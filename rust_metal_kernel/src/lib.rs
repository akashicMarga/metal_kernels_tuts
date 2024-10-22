use metal::*;
use pyo3::prelude::*;
use std::mem;

fn create_device() -> Device {
    Device::system_default().expect("No Metal device found")
}

fn run_kernel(kernel: &Function, input_data: Vec<f32>, output_size: usize) -> Vec<f32> {
    let device = create_device();
    let command_queue = device.new_command_queue();
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(kernel)
        .expect("Failed to create pipeline state");

    // Create input buffer (shared memory between CPU and GPU)
    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (input_data.len() * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create output buffer (shared memory between CPU and GPU)
    let output_buffer = device.new_buffer(
        (output_size * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&input_buffer), 0); // Set input buffer at index 0
    encoder.set_buffer(1, Some(&output_buffer), 0); // Set output buffer at index 1

    let thread_group_size = 256; // Optimal thread group size
    let thread_count = MTLSize::new(output_size as u64, 1, 1);
    let threads_per_group = MTLSize::new(thread_group_size as u64, 1, 1);

    encoder.dispatch_threads(thread_count, threads_per_group);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Map the output buffer's contents back to Rust memory
    let ptr = output_buffer.contents() as *mut f32;
    let mut output_data = vec![0.0; output_size];
    unsafe {
        ptr.copy_to_nonoverlapping(output_data.as_mut_ptr(), output_size);
    }

    output_data
}

fn run_metal_square_kernel(input_data: Vec<f32>) -> Vec<f32> {
    let source = include_str!("square_kernel.metal"); // Load Metal shader source
    let device = create_device();
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("Failed to create Metal library");
    let kernel_function = library
        .get_function("square_kernel", None)
        .expect("Failed to get square kernel function");

    run_kernel(&kernel_function, input_data.clone(), input_data.len())
}

fn run_metal_cube_kernel(input_data: Vec<f32>) -> Vec<f32> {
    let source = include_str!("square_kernel.metal"); // Reuse the same source with cube function
    let device = create_device();
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("Failed to create Metal library");
    let kernel_function = library
        .get_function("cube_kernel", None)
        .expect("Failed to get cube kernel function");

    run_kernel(&kernel_function, input_data.clone(), input_data.len())
}

// Exposing both functions to Python
#[pyfunction]
fn square_numbers(input: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(run_metal_square_kernel(input))
}

#[pyfunction]
fn cube_numbers(input: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(run_metal_cube_kernel(input))
}

// Python module setup
#[pymodule]
fn rust_metal_kernel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(square_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(cube_numbers, m)?)?;
    Ok(())
}
