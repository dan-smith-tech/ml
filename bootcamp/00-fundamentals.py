import torch

# DEF: Tensor - A method of storing multidimensional data (e.g., a matrix is a 2D tensor).

SCALAR = torch.tensor(7)
SCALAR_DIM = SCALAR.ndim
SCALAR_VAL = SCALAR.item()
print(f'Create a scalar: {SCALAR}')
print(f'Dimensions of the scalar: {SCALAR_DIM}')
print(f'Value of the scalar: {SCALAR_VAL}')
print()

VECTOR = torch.tensor([7, 7])
VECTOR_DIM = VECTOR.ndim 
VECTOR_SHAPE = VECTOR.shape
print(f'Create a vector: {VECTOR}')
print(f'Dimensions of the vector: {VECTOR_DIM}')
print(f'Shape of the vector: {VECTOR_SHAPE}')
print()

MATRIX = torch.tensor([[7,8],[9,12]])
MATRIX_DIM = MATRIX.ndim
MATRIX_SHAPE = MATRIX.shape
print(f'Create a matrix: {MATRIX}')
print(f'Dimensions of the matrix: {MATRIX_DIM}')
print(f'Shape of the matrix: {MATRIX_SHAPE}')
print()

TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
TENSOR_DIM = TENSOR.ndim
TENSOR_SHAPE = TENSOR.shape
print(f'Create a tensor: {TENSOR}')
print(f'Dimensions of the tensor: {TENSOR_DIM}')
print(f'Shape of the tensor: {TENSOR_SHAPE}')
print(f'The 0th element of the tensor: {TENSOR[0]}')
print()

# can I create an uneven tensor? (NO)
# print(torch.tensor([[[1,2,3],[4,5],[6]]]))
