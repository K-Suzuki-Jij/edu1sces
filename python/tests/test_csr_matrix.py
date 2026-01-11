from edu1sces.core import CsrMatrix


def test_from_dense() -> None:
    m = CsrMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
    assert m.row_dim == 2
    assert m.col_dim == 2
    assert m.rows == [0, 2, 4]
    assert m.cols == [0, 1, 0, 1]
    assert m.vals == [1.0, 2.0, 3.0, 4.0]


def test_constructor() -> None:
    m = CsrMatrix(2, 2, [0, 2, 4], [0, 1, 0, 1], [1.0, 2.0, 3.0, 4.0])
    assert m.row_dim == 2
    assert m.col_dim == 2
    assert m.rows == [0, 2, 4]
    assert m.cols == [0, 1, 0, 1]
    assert m.vals == [1.0, 2.0, 3.0, 4.0]


def test_mul() -> None:
    # [[2, 0], [0, 3]] * [[1, 2], [3, 4]] = [[2, 4], [9, 12]]
    a = CsrMatrix.from_dense([[2.0, 0.0], [0.0, 3.0]])
    b = CsrMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
    c = a * b
    assert c.row_dim == 2
    assert c.col_dim == 2
    assert c.rows == [0, 2, 4]
    assert c.cols == [0, 1, 0, 1]
    assert c.vals == [2.0, 4.0, 9.0, 12.0]


def test_add() -> None:
    a = CsrMatrix.from_dense([[1.0, 0.0], [0.0, 2.0]])
    b = CsrMatrix.from_dense([[0.0, 3.0], [4.0, 0.0]])
    c = a + b
    assert c.row_dim == 2
    assert c.col_dim == 2
    assert c.rows == [0, 2, 4]
    assert c.cols == [0, 1, 0, 1]
    assert c.vals == [1.0, 3.0, 4.0, 2.0]


def test_sub() -> None:
    a = CsrMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
    b = CsrMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
    c = a - b
    assert c.row_dim == 2
    assert c.col_dim == 2
    assert c.rows == [0, 0, 0]
    assert c.cols == []
    assert c.vals == []


def test_scalar_mul() -> None:
    a = CsrMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
    b = 2.0 * a
    assert b.row_dim == 2
    assert b.col_dim == 2
    assert b.rows == [0, 2, 4]
    assert b.cols == [0, 1, 0, 1]
    assert b.vals == [2.0, 4.0, 6.0, 8.0]
