import rust_metal_kernel

input_data = [1.0, 2.0, 3.0, 4.0]

squared_output = rust_metal_kernel.square_numbers(input_data)
print(f"Squared Output: {squared_output}")

cubed_output = rust_metal_kernel.cube_numbers(input_data)
print(f"Cubed Output: {cubed_output}")
