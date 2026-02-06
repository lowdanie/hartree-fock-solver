from collections import Counter
import dataclasses
from typing import Any
from collections.abc import Sequence
import pytest

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

import slaterform as sf
from tests.jax_utils import block_utils


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class TestNode:
    __test__ = False  # Prevent pytest from collecting this class as a test
    data1: sf.Array
    data2: sf.Array
    label: str

    def tree_flatten(self):
        children = (self.data1, self.data2)
        aux_data = self.label
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls, aux_data: str, children: tuple[jax.Array, ...]
    ) -> "TestNode":
        return cls(data1=children[0], data2=children[1], label=aux_data)


@dataclasses.dataclass
class _ComputeTreeSignatureTestCase:
    tree: Any
    expected_signature: sf.TreeSignature


@dataclasses.dataclass
class _BatchTreeTuplesTestCase:
    trees: list[Any]
    tuple_length: int
    tuple_indices: list[tuple[int, ...]]
    batch_size: int
    expected_batched: Sequence[sf.BatchedTreeTuples]


@dataclasses.dataclass
class _BatchTreeTuplesPropertyTestCase:
    trees: list[Any]
    tuple_length: int
    tuple_indices: list[tuple[int, ...]]
    batch_size: int


def _assert_trees_equal(actual: Any, expected: Any):
    """Asserts that two pytrees are equal."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_array_equal(x, y), actual, expected
    )


@pytest.mark.parametrize(
    "case",
    [
        _ComputeTreeSignatureTestCase(
            tree=TestNode(
                data1=jnp.array([1.0, 2.0, 3.0]),
                data2=jnp.array([[1, 2], [3, 4]]),
                label="node1",
            ),
            expected_signature=sf.TreeSignature(
                aux_data=jax.tree.structure(
                    TestNode(
                        data1=jnp.array([1.0, 2.0, 3.0]),
                        data2=jnp.array([[1, 2], [3, 4]]),
                        label="node1",
                    )
                ),
                leaf_shapes=((3,), (2, 2)),
            ),
        ),
    ],
)
def test_compute_tree_signature(case: _ComputeTreeSignatureTestCase):
    signature = sf.jax_utils.compute_tree_signature(case.tree)
    assert signature == case.expected_signature


@pytest.mark.parametrize(
    "case",
    [
        _BatchTreeTuplesTestCase(
            trees=[
                TestNode(
                    data1=jnp.array([1.0]),
                    data2=jnp.array([1.0]),
                    label="nodeA",
                ),
            ],
            tuple_length=2,
            tuple_indices=[],
            batch_size=2,
            expected_batched=[],
        ),
        _BatchTreeTuplesTestCase(
            trees=[
                # Group: A1
                TestNode(
                    data1=jnp.array([1.0]),
                    data2=jnp.array([1.0]),
                    label="nodeA",
                ),
                # Group: A1
                TestNode(
                    data1=jnp.array([2.0]),
                    data2=jnp.array([2.0]),
                    label="nodeA",
                ),
                # Group: A2
                TestNode(
                    data1=jnp.array([1.0, 2.0]),
                    data2=jnp.array([1.0, 2.0]),
                    label="nodeA",
                ),
                # Group: B1
                TestNode(
                    data1=jnp.array([1.0]),
                    data2=jnp.array([1.0]),
                    label="nodeB",
                ),
            ],
            tuple_length=2,
            tuple_indices=[
                (0, 0),  # A1, A1
                (0, 1),  # A1, A1
                (0, 2),  # A1, A2
                (0, 3),  # A1, B1
                (1, 1),  # A1, A1
                (1, 2),  # A1, A2
                (1, 3),  # A1, B1
                (2, 2),  # A2, A2
                (2, 3),  # A2, B1
                (3, 3),  # B1, B1
            ],
            batch_size=2,
            expected_batched=[
                # A1, A1
                sf.BatchedTreeTuples(
                    stacks=(
                        TestNode(
                            data1=jnp.array([[1.0], [2.0]]),
                            data2=jnp.array([[1.0], [2.0]]),
                            label="nodeA",
                        ),
                        TestNode(
                            data1=jnp.array([[1.0], [2.0]]),
                            data2=jnp.array([[1.0], [2.0]]),
                            label="nodeA",
                        ),
                    ),
                    global_tree_indices=(
                        jnp.array([0, 1]),
                        jnp.array([0, 1]),
                    ),
                    tuple_indices=jnp.array(
                        [
                            [
                                [0, 0],
                                [0, 1],
                            ],
                            [
                                [1, 1],
                                [0, 0],  # padding
                            ],
                        ]
                    ),
                    padding_mask=jnp.array(
                        [
                            [1, 1],
                            [1, 0],
                        ]
                    ),
                ),
                # A1, A2
                sf.BatchedTreeTuples(
                    stacks=(
                        TestNode(
                            data1=jnp.array([[1.0], [2.0]]),
                            data2=jnp.array([[1.0], [2.0]]),
                            label="nodeA",
                        ),
                        TestNode(
                            data1=jnp.array([[1.0, 2.0]]),
                            data2=jnp.array([[1.0, 2.0]]),
                            label="nodeA",
                        ),
                    ),
                    global_tree_indices=(
                        jnp.array([0, 1]),
                        jnp.array([2]),
                    ),
                    tuple_indices=jnp.array(
                        [
                            [
                                [0, 0],
                                [1, 0],
                            ],
                        ]
                    ),
                    padding_mask=jnp.array(
                        [
                            [1, 1],
                        ]
                    ),
                ),
                # A1, B1
                sf.BatchedTreeTuples(
                    stacks=(
                        TestNode(
                            data1=jnp.array([[1.0], [2.0]]),
                            data2=jnp.array([[1.0], [2.0]]),
                            label="nodeA",
                        ),
                        TestNode(
                            data1=jnp.array([[1.0]]),
                            data2=jnp.array([[1.0]]),
                            label="nodeB",
                        ),
                    ),
                    global_tree_indices=(
                        jnp.array([0, 1]),
                        jnp.array([3]),
                    ),
                    tuple_indices=jnp.array(
                        [
                            [
                                [0, 0],
                                [1, 0],
                            ],
                        ]
                    ),
                    padding_mask=jnp.array(
                        [
                            [1, 1],
                        ]
                    ),
                ),
                # A2, A2
                sf.BatchedTreeTuples(
                    stacks=(
                        TestNode(
                            data1=jnp.array([[1.0, 2.0]]),
                            data2=jnp.array([[1.0, 2.0]]),
                            label="nodeA",
                        ),
                        TestNode(
                            data1=jnp.array([[1.0, 2.0]]),
                            data2=jnp.array([[1.0, 2.0]]),
                            label="nodeA",
                        ),
                    ),
                    global_tree_indices=(
                        jnp.array([2]),
                        jnp.array([2]),
                    ),
                    tuple_indices=jnp.array(
                        [
                            [
                                [0, 0],
                                [0, 0],  # padding
                            ],
                        ]
                    ),
                    padding_mask=jnp.array(
                        [
                            [1, 0],
                        ]
                    ),
                ),
                # A2, B1
                sf.BatchedTreeTuples(
                    stacks=(
                        TestNode(
                            data1=jnp.array([[1.0, 2.0]]),
                            data2=jnp.array([[1.0, 2.0]]),
                            label="nodeA",
                        ),
                        TestNode(
                            data1=jnp.array([[1.0]]),
                            data2=jnp.array([[1.0]]),
                            label="nodeB",
                        ),
                    ),
                    global_tree_indices=(
                        jnp.array([2]),
                        jnp.array([3]),
                    ),
                    tuple_indices=jnp.array(
                        [
                            [[0, 0], [0, 0]],  # padding
                        ]
                    ),
                    padding_mask=jnp.array(
                        [
                            [1, 0],
                        ]
                    ),
                ),
                # B1, B1
                sf.BatchedTreeTuples(
                    stacks=(
                        TestNode(
                            data1=jnp.array([[1.0]]),
                            data2=jnp.array([[1.0]]),
                            label="nodeB",
                        ),
                        TestNode(
                            data1=jnp.array([[1.0]]),
                            data2=jnp.array([[1.0]]),
                            label="nodeB",
                        ),
                    ),
                    global_tree_indices=(
                        jnp.array([3]),
                        jnp.array([3]),
                    ),
                    tuple_indices=jnp.array(
                        [
                            [
                                [0, 0],
                                [0, 0],  # padding
                            ],
                        ]
                    ),
                    padding_mask=jnp.array(
                        [
                            [1, 0],
                        ]
                    ),
                ),
            ],
        ),
    ],
)
def test_batch_tree_tuples(case: _BatchTreeTuplesTestCase):
    batched_tree_tuples = sf.jax_utils.batch_tree_tuples(
        trees=case.trees,
        tuple_length=case.tuple_length,
        tuple_indices=case.tuple_indices,
        batch_size=case.batch_size,
    )

    assert len(batched_tree_tuples) == len(case.expected_batched)
    for actual, expected in zip(batched_tree_tuples, case.expected_batched):
        _assert_trees_equal(actual, expected)


_BatchTreeTuplePropertyTestCases = [
    _BatchTreeTuplesPropertyTestCase(
        trees=[
            TestNode(
                data1=jnp.array([1.0]),
                data2=jnp.array([1.0]),
                label="nodeA",
            ),
            TestNode(
                data1=jnp.array([2.0]),
                data2=jnp.array([2.0]),
                label="nodeA",
            ),
            TestNode(
                data1=jnp.array([1.0, 2.0]),
                data2=jnp.array([1.0, 2.0]),
                label="nodeA",
            ),
            TestNode(
                data1=jnp.array([1.0]),
                data2=jnp.array([1.0]),
                label="nodeB",
            ),
            TestNode(
                data1=jnp.array([3.0]),
                data2=jnp.array([3.0]),
                label="nodeB",
            ),
        ],
        tuple_length=4,
        tuple_indices=[
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (2, 2, 2, 2),
            (3, 3, 3, 3),
            (4, 4, 4, 4),
            (0, 1, 0, 1),
            (1, 2, 1, 2),
            (2, 3, 2, 3),
            (3, 4, 3, 4),
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (2, 3, 4, 0),
            (3, 4, 0, 1),
            (4, 0, 1, 2),
        ],
        batch_size=2,
    ),
]


def _unstack_tree(stacked_tree: Any) -> list[Any]:
    leaves, treedef = jax.tree.flatten(stacked_tree)

    if not leaves:
        return []

    batch_size = leaves[0].shape[0]

    unstacked_trees = []
    for i in range(batch_size):
        sliced_leaves = [leaf[i] for leaf in leaves]
        tree_i = treedef.unflatten(sliced_leaves)
        unstacked_trees.append(tree_i)

    return unstacked_trees


def _validate_global_indices(
    batched_tree: sf.BatchedTreeTuples,
    global_trees: Sequence[TestNode],
):
    local_tree_cols = [_unstack_tree(stack) for stack in batched_tree.stacks]

    for global_indices, local_trees in zip(
        batched_tree.global_tree_indices, local_tree_cols
    ):
        for local_idx, global_idx in enumerate(global_indices):
            _assert_trees_equal(
                local_trees[local_idx], global_trees[global_idx]
            )


@pytest.mark.parametrize("case", _BatchTreeTuplePropertyTestCases)
def test_batch_tree_tuples_preserves_tuples(
    case: _BatchTreeTuplesPropertyTestCase,
):
    batched_tree_tuples = sf.jax_utils.batch_tree_tuples(
        trees=case.trees,
        tuple_length=case.tuple_length,
        tuple_indices=case.tuple_indices,
        batch_size=case.batch_size,
    )

    # Check that the batched tree tuples reconstruct the original tuples.
    reconstructed_tuples = []
    for bt in batched_tree_tuples:
        for global_batch in block_utils.get_global_tuple_batches(bt):
            reconstructed_tuples.extend(global_batch)

    assert Counter(reconstructed_tuples) == Counter(case.tuple_indices)


@pytest.mark.parametrize("case", _BatchTreeTuplePropertyTestCases)
def test_batch_tree_tuples_verify_global_indices(
    case: _BatchTreeTuplesPropertyTestCase,
):
    batched_tree_tuples = sf.jax_utils.batch_tree_tuples(
        trees=case.trees,
        tuple_length=case.tuple_length,
        tuple_indices=case.tuple_indices,
        batch_size=case.batch_size,
    )

    for bt in batched_tree_tuples:
        _validate_global_indices(bt, case.trees)


@pytest.mark.parametrize("case", _BatchTreeTuplePropertyTestCases)
def test_batch_tree_tuples_batch_sizes(
    case: _BatchTreeTuplesPropertyTestCase,
):
    batched_tree_tuples = sf.jax_utils.batch_tree_tuples(
        trees=case.trees,
        tuple_length=case.tuple_length,
        tuple_indices=case.tuple_indices,
        batch_size=case.batch_size,
    )

    for bt in batched_tree_tuples:
        batch_size = bt.tuple_indices.shape[1]
        assert batch_size == case.batch_size
