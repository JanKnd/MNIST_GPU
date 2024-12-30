/*
for dense_layer:
input dimensions: input_size x 1
weights dimensions: output_size x input_size
biases dimensions: output_size x 1
output dimensions: output_size x 1


1.forward
    loop:
    1. fw dense
    2. fw activation
    fw mse
2.backward
    bw mse
    loop:
    1. bw actibation
    2. bw weights
    3. bw biases
    4. bw input
3.update
    add element wise grad*lr to values
*/

pub mod data;
pub mod network;
pub mod run;
pub mod train;

use rand::Rng;
use std::{mem::size_of_val, str::FromStr, time::Instant};
use wgpu::util::DeviceExt;

#[cfg_attr(test, allow(dead_code))]
async fn run() {
    let start0 = Instant::now();
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = vec![1., 0., 1., 0., 0., 1., 0., 1., 0., 0.];
    //let values: Vec<f32> = (0..5).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let grad: Vec<f32> = vec![0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5];
    let dims: Vec<u32> = vec![0, 2, 1, 1, 2, 2, 2, 1, 6, 2, 1, 1, 8, 2, 1, 1];
    let status: Vec<f32> = vec![0.];
    let duration0 = start0.elapsed();
    print!(
        "Time elapsed in creating values and dims is: {:?} \n",
        duration0
    );

    let start1 = Instant::now();
    let res = execute_gpu(&values, &grad, &dims, &status).await.unwrap();
    let duration1 = start1.elapsed();
    println!("Time elapsed in execute_gpu() is: {:?}", duration1);
    print!("{:?}", res);
}

#[cfg_attr(test, allow(dead_code))]
async fn execute_gpu(
    values: &[f32],
    grad: &[f32],
    dims: &[u32],
    status: &Vec<f32>,
) -> Option<Vec<f32>> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, values, grad, dims, status).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    grad: &[f32],
    dims: &[u32],
    status: &[f32],
) -> Option<Vec<f32>> {
    // Load the shaders from WGSL
    //Dense layer shaders
    let dense_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_forward.wgsl"));
    let dense_input_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_input_backward.wgsl"));
    let dense_weight_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_weights_backward.wgsl"));
    let dense_bias_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_biases_backward.wgsl"));
    //Activation layer shaders
    let activation_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/activation_fn_forward.wgsl"));
    let activation_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/activation_fn_backward.wgsl"));
    //MSE shaders
    let mse_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/mse_forward.wgsl"));
    let mse_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/mse_backward.wgsl"));

    // Get the size in bytes of the buffer
    let size_values = size_of_val(values) as wgpu::BufferAddress;
    let size_grad = size_of_val(grad) as wgpu::BufferAddress;
    let size_dims = size_of_val(dims) as wgpu::BufferAddress;
    let size_status = size_of_val(status) as wgpu::BufferAddress;

    // Create staging buffers for transferring data from GPU to CPU
    let staging_buffer_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_values,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer_grad = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_grad,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer_dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_dims,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer_status = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_status,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create buffers for values, grad, dims and status
    let values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Values Buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let grad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gradient Buffer"),
        contents: bytemuck::cast_slice(grad),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dimensions Buffer"),
        contents: bytemuck::cast_slice(dims),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let status_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Status Buffer"),
        contents: bytemuck::cast_slice(status),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dims_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: status_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline layout
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipelines for both shaders
    let compute_pipeline_a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline A"),
        layout: Some(&compute_pipeline_layout),
        module: &dense_forward_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let compute_pipeline_b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline B"),
        layout: Some(&compute_pipeline_layout),
        module: &dense_input_backward_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Execute the first shader
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass A"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline_a);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(2, 1, 1);
    }

    // Execute the second shader
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass B"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline_b);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(2, 1, 1);
    }

    // Copy data from GPU buffer to staging buffer
    encoder.copy_buffer_to_buffer(&values_buffer, 0, &staging_buffer_values, 0, size_values);
    encoder.copy_buffer_to_buffer(&grad_buffer, 0, &staging_buffer_grad, 0, size_values);
    encoder.copy_buffer_to_buffer(&dims_buffer, 0, &staging_buffer_dims, 0, size_dims);
    encoder.copy_buffer_to_buffer(&status_buffer, 0, &staging_buffer_status, 0, size_status);

    // Submit command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Map staging buffer to CPU address space
    let values_buffer_slice = staging_buffer_values.slice(..);
    let (sender, receiver) = flume::bounded(1);
    values_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device to ensure the mapping is complete
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Read data from staging buffer
    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = values_buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer_values.unmap();
        //Some(result)
    } else {
        // panic!("failed to run compute on gpu!")
    }

    let grad_buffer_slice = staging_buffer_grad.slice(..);
    let (sender, receiver) = flume::bounded(1);
    grad_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device to ensure the mapping is complete
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Read data from staging buffer
    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = grad_buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer_grad.unmap();
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}

//#[cfg(test)]
//mod tests;
