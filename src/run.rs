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
pub async fn run_on_example(dnn: Network, training_set: Data) -> Network{
    let mut network = dnn;

    for i in 0..training_set.dims.len() / 8 {
        print!("Example: {:?} \n", i);
        let start = Instant::now();


        // Print status before running GPU
        println!("Status before run_gpu: {:?}", network.status);

        network = run_gpu(
            &network.values,
            &network.grad,
            &network.dims,
            &network.status,
            &training_set.values,
            &training_set.dims,
        )
            .await
            .unwrap();

        // Print status after running GPU
        println!("Status after run_gpu: {:?}", network.status);

        let duration = start.elapsed();
        println!("Time elapsed in execute_gpu() is: {:?}", duration);

        let mse = network.status[3] / network.status[4];
        network.status[3] = 0.;
        network.status[4] = 0.;

        dbg!(&network.values);
        print!("{:?} \n MSE: {:?} \n \n ", network.status, mse);

        //number of current example
        let example = network.status[2] as usize;
        //length of the output matrix
        let output_size = training_set.dims[example * 8 + 5] as usize * training_set.dims[example * 4 + 6] as usize;
        //length of the input matrix
        let input_size = training_set.dims[example * 8 + 1] as usize * training_set.dims[example * 8 + 2] as usize;
        //input matrix
        let input = &training_set.values[training_set.dims[example * 8] as usize..training_set.dims[example * 8] as usize + input_size];

        let network_pred = &network.values[network.values.len() - output_size..];
        let target = &training_set.values[training_set.dims[example * 8 + 4] as usize..training_set.dims[example * 8 + 4] as usize + output_size];
        println!("Input:      {:?}, \n Prediction: {:?} \n Target:     {:?} \n", input ,network_pred, target);
        network.status[2] = i as f32;
        network.status[0] = 0.;
    }

    network
}

#[cfg_attr(test, allow(dead_code))]
async fn run_gpu(
    values: &[f32],
    grad: &[f32],
    dims: &[u32],
    status: &[f32],
    data_values: &[f32],
    data_dims: &[u32],
) -> Option<Network> {
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

    run_gpu_inner(
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

async fn run_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    grad: &[f32],
    dims: &[u32],
    status: &[f32],
    data_values: &[f32],
    data_dims: &[u32],
) -> Option<Network> {
    // Load the shaders from WGSL

    //Dense layer shaders
    let dense_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/dense_forward.wgsl"));

    //Activation layer shaders
    let activation_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/activation_fn_forward.wgsl"));

    //MSE shaders
    let mse_forward_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/mse_forward.wgsl"));

    //other shaders
    let input_setter_shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/input_setter.wgsl"));


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
    let bind_group_layout_network = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Network Bind Group Layout"),
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

    let bind_group_layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Data Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false},
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
        ],
    });

    // Create bind group
    let bind_group_network = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Network Bind Group"),
        layout: &bind_group_layout_network,
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

    let bind_group_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Data Bind Group"),
        layout: &bind_group_layout_data,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data_values_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_dims_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline layout
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout_network, &bind_group_layout_data],
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

    let cp_activation_fn_forward =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline Activation Forward"),
            layout: Some(&compute_pipeline_layout),
            module: &activation_forward_shader,
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

    let cp_input_setter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline Input Setter"),
        layout: Some(&compute_pipeline_layout),
        module: &input_setter_shader,
        entry_point: Option::from("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Update the status buffer with the new value of status[2]
    queue.write_buffer(&status_buffer, 0, bytemuck::cast_slice(&status));

    // Create command encoder
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    //shader execution
    //runs shaders for each example of the training set

    //data_dims.len() / 2 * 4 as each example has 2 matrices with each 4 values
    {
        //runs input setter shader
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Input Setter"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&cp_input_setter);
            cpass.set_bind_group(0, &bind_group_network, &[]);
            cpass.set_bind_group(1, &bind_group_data, &[]);
            cpass.dispatch_workgroups(dims[1] * dims[2], 1, 1);
        }

        //dims.len() / 4 * 4 as each layer has 4 matrices with each 4 values
        for j in 0..dims.len() / (4 * 4) {
            // runs dense layer forward pass shader
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dense Forward Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&cp_dense_forward);
                cpass.set_bind_group(0, &bind_group_network, &[]);
                cpass.set_bind_group(1, &bind_group_data, &[]);
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
                cpass.set_bind_group(0, &bind_group_network, &[]);
                cpass.set_bind_group(1, &bind_group_data, &[]);
                //because the output of the activation function is the same size as the input, same indexes are used
                cpass.dispatch_workgroups(dims[j * 4 + 12 + 1] * dims[j * 4 + 12 + 2], 1, 1);
            }
        }


        // runs mse forward pass shader
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MSE Forward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&cp_mse_forward);
            cpass.set_bind_group(0, &bind_group_network, &[]);
            cpass.set_bind_group(1, &bind_group_data, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
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

    // read data from staging buffer
    let values: Vec<f32>;
    {
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
            values = result;
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    let grad: Vec<f32>;
    {
        let grad_buffer_slice = staging_buffer_grad.slice(..);
        let (sender, receiver) = flume::bounded(1);
        grad_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device to ensure the mapping is complete
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Read data from staging buffer
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = grad_buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer_grad.unmap();
            grad = result;
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    let dims: Vec<u32>;
    {
        let dims_buffer_slice = staging_buffer_dims.slice(..);
        let (sender, receiver) = flume::bounded(1);
        dims_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device to ensure the mapping is complete
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Read data from staging buffer
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = dims_buffer_slice.get_mapped_range();
            let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer_dims.unmap();
            dims = result;
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    let mut status: Vec<f32>;
    {
        let status_buffer_slice = staging_buffer_status.slice(..);
        let (sender, receiver) = flume::bounded(1);
        status_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device to ensure the mapping is complete
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Read data from staging buffer
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = status_buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer_status.unmap();
            status = result;
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    /*


    let status_buffer_slice = staging_buffer_status.slice(..);
    let (sender, receiver) = flume::bounded(1);
    status_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device to ensure the mapping is complete
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Read data from staging buffer
    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = status_buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer_status.unmap();
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }

     */

    Some(Network{
        values,
        grad,
        dims,
        status,
    })
}

