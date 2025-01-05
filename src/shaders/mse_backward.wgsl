@group(0)
@binding(0)
var<storage, read_write> values: array<f32>;
//[i,n,p,u,t,w,e,i,g,h,t,s,b,i,a,s,e,s,i,n,p,u,t,...]

@group(0)
@binding(1)
var<storage, read_write> grad: array<f32>;
//same as values only for gradients

@group(0)
@binding(2)
var<storage, read_write> dims: array<u32>;
//[index in values, x, y, z, index in values, x, y, z, ...]

@group(0)
@binding(3)
var<storage, read_write> status: array<f32>;
//status[0] = number of current layer

@group(1)
@binding(0)
var<storage, read_write> data_values: array<f32>;

@group(1)
@binding(1)
var<storage, read_write> data_dims: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    var output_mat_start_index = dims[(u32(status[0]) * 4u) * 4u];

    var output_length = dims[(u32(status[0]) * 4u) * 4u + 1u] * dims[(u32(status[0]) * 4u) * 4u + 2u];

    var example_output_start_index = data_dims[u32(status[2]) * 2u * 4u + 4u];

    grad[global_id.x  + output_mat_start_index] =  ((2.0 / f32(output_length)) * (values[global_id.x + output_mat_start_index] - data_values[global_id.x + example_output_start_index])) ;

    if (global_id.x == output_length - 1u) {
            status[2] += 1.0;
            status[0] -= 1.0;
    }
}
