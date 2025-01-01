@group(0)
@binding(0)
var<storage, read_write> values: array<f32>;
//[i,n,p,u,t,w,e,i,g,h,t,s,b,i,a,s,e,s,i,n,p,u,t,...]

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
    var output_mat_id = u32(status[0]) * 4u - 1u;
    var output_length = dims[output_mat_id * 4u + 1u] * dims[output_mat_id * 4u + 2u];

    var output_mat_start_index = dims[output_mat_id * 4u];

    var example_output_start_index = data_dims[u32(status[2]) * 2u * 4u];

    var sum = 0.0;
    for(var i: u32 = 0u; i < output_length; i = i + 1u) {
        var base = values[i + output_mat_start_index] - data_values[i + example_output_start_index];
        sum += base * base;
    }
    status[3] += (sum / f32(output_length));
    status[4] += 1.0;
}
