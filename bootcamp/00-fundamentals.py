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

# Random tensors - Important for initialisation
random_tensor = torch.rand(3, 4)
print(f'Random tensor: {random_tensor}')

# Create random tensor to similar size as image
random_image_tensor = torch.rand(size=(3, 224, 224)) # height, width, RGB
print(random_image_tensor)

# NOTE: size and shape return/work the same

# Create a tensor of zeros

zeros = torch.zeros(3,4)
print(zeros)

ones = torch.ones(3,4)
print(ones)

# dtype is 'default type'
print(f'Default data type: {ones.dtype}')

# Create a range of tensors
tensor_range = torch.arange(start=0, end=1000, step=77)
print(tensor_range)

# Create tensors 'like' another (e.g., same shape)
tensor_like = torch.zeros_like(input=tensor_range)
print(tensor_like)
print()

float_32_tensor = torch.tensor([3.0,6.0], dtype=None, # data type (32 bit float, etc.)
                                       device=None, # CPU/GPU/etc. - tensors need to exist on the same device in order to interact
                                       requires_grad=False # whether or not to track the gradient with operations on the tensor
)
print(f'Float 32 tensor: {float_32_tensor.dtype}')
# using smaller data types (e.g., 16-bit) will make computations faster, at the cost of precision
float_16_tensor = float_32_tensor.type(torch.float16)
print(f'Float16 tensor: {float_16_tensor.dtype}');
print(f'When multiplying float16 by float32, the result is 32: {(float_16_tensor * float_32_tensor).dtype}')
print()

# Manipulating tensors
tensor_to_maniuplate = torch.tensor([1,2,3])
print(f'Add 10 to {tensor_to_maniuplate}: {tensor_to_maniuplate + 10}')
print(f'Multiply {tensor_to_maniuplate} by 10: {tensor_to_maniuplate * 10}')
# ...etc...
print()

# Matrix multiplication
print(tensor_to_maniuplate, '*', tensor_to_maniuplate, ' is ', tensor_to_maniuplate * tensor_to_maniuplate)
print(tensor_to_maniuplate, ' dot ', tensor_to_maniuplate, ' is ', torch.matmul(tensor_to_maniuplate, tensor_to_maniuplate))
# Shape errors
# 1. Inner dimensions must match:
#   (row height must match column height)
#   - (3,2) @ (3,2) WON'T work
#   - (2,3) @ (3,2) WILL work
#   - (3,2) @ (2,3) WILL work
# 2. The resulting matrix has the shape of the outer dimensions:
#   (each row applied to each column)
#   - (2,3) @ (3,2) -> (2,2)
print()

# Transposing to fix shape issues
tensor_A = torch.tensor([[1,2],[3,4],[5,6]])
tensor_B = torch.tensor([[7,10],[8,11],[9,12]])
print(f'Cannot matmul {tensor_A} and {tensor_B} as their inner dimensions do not match: {tensor_A.shape, tensor_B.shape}') 
# transpose switches the dimensions/axes of a given tensor
print(f'Transpose a matrix to switch their axes and fix the incompatibility: {tensor_A, tensor_B.T}')
print(f'So now they can be matrix multiplied: {torch.mm(tensor_A, tensor_B.T)}')
print('NOTE: the same information is represented in these tensors, it has simply been rearranged in one of them.')
