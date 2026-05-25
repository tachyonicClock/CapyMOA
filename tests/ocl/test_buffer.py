import torch

from capymoa.ocl.util._buffer import BufferList, BufferDict


def test_buffer_list_mutations_preserve_order_and_identity():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.tensor([3.0])
    d = torch.tensor([4.0])

    buffers = BufferList([a, c])

    assert len(buffers) == 2
    assert buffers[0] is a
    assert buffers[1] is c
    assert a in buffers
    assert torch.tensor([1.0]) not in buffers

    buffers.insert(1, b)

    assert len(buffers) == 3
    assert buffers[0] is a
    assert buffers[1] is b
    assert buffers[2] is c

    del buffers[1]

    assert len(buffers) == 2
    assert buffers[0] is a
    assert buffers[1] is c
    assert list(buffers._buffers.keys()) == ["0", "1"]

    buffers[1] = d

    assert buffers[1] is d
    assert c not in buffers


def test_buffer_dict_preserves_insertion_order_and_identity():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.tensor([3.0])

    buffers = BufferDict({"a": a, "b": b, "c": c})

    assert list(buffers.keys()) == ["a", "b", "c"]
    assert list(iter(buffers)) == ["a", "b", "c"]
    assert buffers["a"] is a
    assert buffers["b"] is b
    assert buffers["c"] is c


def test_buffer_dict_mutations_follow_ordered_dict_rules():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.tensor([3.0])
    d = torch.tensor([4.0])

    buffers = BufferDict({"a": a, "b": b, "c": c})
    assert list(buffers.items()) == [("a", a), ("b", b), ("c", c)]

    buffers["b"] = d
    assert list(buffers.keys()) == ["a", "b", "c"]
    assert buffers["b"] is d
    assert buffers._buffers["b"] is d

    del buffers["a"]
    assert list(buffers.keys()) == ["b", "c"]
    assert "a" not in buffers
    assert "a" not in buffers._buffers

    buffers["a"] = a
    assert list(buffers.keys()) == ["b", "c", "a"]
    assert list(buffers._buffers.keys()) == ["b", "c", "a"]


def test_buffer_dict_contains_checks_keys_not_tensor_values():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    buffers = BufferDict({"a": a})

    assert "a" in buffers
    assert "b" not in buffers
    assert a not in buffers

    buffers["b"] = b
    assert "b" in buffers
