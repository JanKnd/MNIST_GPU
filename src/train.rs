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

status[0] = current layers
status[1] = learning rate
status[2] = current training set
status[3] = mse added up
status[4] = mse count

*/

use std::time::Instant;
use wgpu::util::DeviceExt;

use crate::data::Data;
use crate::network::Network;

#[cfg_attr(test, allow(dead_code))]
async fn run_train() {
    let network = Network::new_xor();
    let data = Data::new_xor();

    let start = Instant::now();
    let res = train_gpu(
        &network.values,
        &network.grad,
        &network.dims,
        &network.status,
        &data.values,
        &data.dims,
    )
    .await
    .unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in execute_gpu() is: {:?}", duration);
    print!("{:?}", res);
}

#[cfg_attr(test, allow(dead_code))]
async fn train_gpu(
    values: &[f32],
    grad: &[f32],
    dims: &[u32],
    status: &[f32],
    data_values: &[f32],
    data_dims: &[u32],
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

    train_gpu_inner(
        &device,
        &queue,
        values,
        grad,
        dims,
        status,
        data_values,
        data_dims,
    )
    .await
}

async fn train_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    grad: &[f32],
    dims: &[u32],
    status: &[f32],
    data_values: &[f32],
    data_dims: &[u32],
) -> Option<Vec<f32>> {
    // Load the shaders from WGSL

    //Dense layer shaders
    let dense_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_forward.wgsl"));

    let dense_input_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_input_backward.wgsl"));

    let dense_weights_backward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_weights_backward.wgsl"));

    let dense_biases_backward_shader =
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

    //other shaders
    let input_setter_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/input_setter.wgsl"));

    let apply_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/apply_gradient.wgsl"));

    // Get the size in bytes of the buffer
    let size_values = size_of_val(values) as wgpu::BufferAddress;
    let size_grad = size_of_val(grad) as wgpu::BufferAddress;
    let size_dims = size_of_val(dims) as wgpu::BufferAddress;
    let size_status = size_of_val(status) as wgpu::BufferAddress;
    let size_data_values = size_of_val(data_values) as wgpu::BufferAddress;
    let size_data_dims = size_of_val(data_dims) as wgpu::BufferAddress;

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

    let staging_buffer_data_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_data_values,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer_data_dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_data_dims,
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

    let data_values_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Data Values Buffer"),
        contents: bytemuck::cast_slice(data_values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let data_dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Data Dimensions Buffer"),
        contents: bytemuck::cast_slice(data_dims),
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
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: data_values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: data_dims_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline layout
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipelines for all shaders
    let cp_dense_forward = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline Dense Forward"),
        layout: Some(&compute_pipeline_layout),
        module: &dense_forward_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let cp_dense_input_backward =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline Dense Input Backward"),
            layout: Some(&compute_pipeline_layout),
            module: &dense_input_backward_shader,
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let cp_dense_weights_backward =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline Dense Weights Backward"),
            layout: Some(&compute_pipeline_layout),
            module: &dense_weights_backward_shader,
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let cp_dense_biases_backward =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline Dense Weights Backward"),
            layout: Some(&compute_pipeline_layout),
            module: &dense_biases_backward_shader,
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let cp_activation_fn_forward =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline Activation Forward"),
            layout: Some(&compute_pipeline_layout),
            module: &activation_forward_shader,
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let cp_activation_fn_backward =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline Activation Backward"),
            layout: Some(&compute_pipeline_layout),
            module: &activation_backward_shader,
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let cp_mse_forward = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline MSE Forward"),
        layout: Some(&compute_pipeline_layout),
        module: &mse_forward_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let cp_mse_backward = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline MSE Backward"),
        layout: Some(&compute_pipeline_layout),
        module: &mse_backward_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let cp_input_setter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline Input Setter"),
        layout: Some(&compute_pipeline_layout),
        module: &input_setter_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let cp_apply_grad = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline Apply Gradient"),
        layout: Some(&compute_pipeline_layout),
        module: &apply_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    //shader execution
    //runs shaders for each example of the training set

    //data_dims.len() / 2 * 4 as each example has 2 matrices with each 4 values
    for i in 0..data_dims.len() / 2 * 4 {
        //runs input setter shader
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Input Setter"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&cp_input_setter);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(dims[1] * dims[2], 1, 1);
        }

        //dims.len() / 4 * 4 as each layer has 4 matrices with each 4 values
        for j in 0..dims.len() / 4 * 4 {
            // runs dense layer forward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dense Forward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_dense_forward);
                cpass.set_bind_group(0, &bind_group, &[]);

                //dims[j * 4 + 12 + 1] = output_rows, dims[j * 4 + 12 + 2] = output_cols
                cpass.dispatch_workgroups(dims[j * 4 + 12 + 1] * dims[j * 4 + 12 + 2], 1, 1);
            }

            // runs activation function forward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Activation Forward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_activation_fn_forward);
                cpass.set_bind_group(0, &bind_group, &[]);
                //because the output of the activation function is the same size as the input, same indexes are used
                cpass.dispatch_workgroups(dims[j * 4 + 12 + 1] * dims[j * 4 + 12 + 2], 1, 1);
            }
        }

        // runs mse forward pass shader
        // TODO: write mse shader first to know workgroup size
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MSE Forward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&cp_mse_forward);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        //Backward pass

        // runs mse backward pass shader
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MSE Backward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&cp_mse_backward);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(dims[dims.len() - 3] * dims[dims.len() - 2], 1, 1);
        }

        for j in (0..dims.len() / 4 * 4).rev() {
            // runs activation function backward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Activation Backward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_activation_fn_backward);
                cpass.set_bind_group(0, &bind_group, &[]);
                //because the output of the activation function is the same size as the input, same indexes are used
                cpass.dispatch_workgroups(dims[j * 4 + 12 + 1] * dims[j * 4 + 12 + 2], 1, 1);
            }

            // runs dense weights backward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dense Weights Backward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_dense_weights_backward);
                cpass.set_bind_group(0, &bind_group, &[]);
                // dims[j * 4 + 4 + 1] = weights_rows, dims[j * 4 + 4 + 2] = weights_cols
                cpass.dispatch_workgroups(dims[j * 4 + 4 + 1] * dims[j * 4 + 4 + 2], 1, 1);
            }

            // runs dense biases backward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dense Biases Backward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_dense_biases_backward);
                cpass.set_bind_group(0, &bind_group, &[]);
                // dims[j * 4 + 8 + 1] = biases_rows, dims[j * 4 + 8 + 2] = biases_cols
                cpass.dispatch_workgroups(dims[j * 4 + 8 + 1] * dims[j * 4 + 8 + 2], 1, 1);
            }

            // runs dense input backward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dense Input Backward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_dense_input_backward);
                cpass.set_bind_group(0, &bind_group, &[]);
                // dims[j * 4 + 1] = output_rows, dims[j * 4 + 2] = output_cols
                cpass.dispatch_workgroups(dims[j * 4 + 1] * dims[j * 4 + 2], 1, 1);
            }
        }

        //runs apply gradient shader
        //TODO think of an efficient way to use multiple dimensions for the workgroups
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Gradient"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&cp_apply_grad);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(values.len() as u32, 1, 1);
        }
    }

    // Copy data from GPU buffer to staging buffer
    encoder.copy_buffer_to_buffer(&values_buffer, 0, &staging_buffer_values, 0, size_values);
    encoder.copy_buffer_to_buffer(&grad_buffer, 0, &staging_buffer_grad, 0, size_values);
    encoder.copy_buffer_to_buffer(&dims_buffer, 0, &staging_buffer_dims, 0, size_dims);
    encoder.copy_buffer_to_buffer(&status_buffer, 0, &staging_buffer_status, 0, size_status);
    encoder.copy_buffer_to_buffer(
        &data_values_buffer,
        0,
        &staging_buffer_data_values,
        0,
        size_data_values,
    );
    encoder.copy_buffer_to_buffer(
        &data_dims_buffer,
        0,
        &staging_buffer_data_dims,
        0,
        size_data_dims,
    );

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
