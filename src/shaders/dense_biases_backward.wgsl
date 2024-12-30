@group(0)
@binding(0)
var<storage, read_write> values: array<f32>;
//[i,n,p,u,t,w,e,i,g,h,t,s,b,i,a,s,e,s,i,n,p,u,t,...]

@group(0)
@binding(1)
var<storage, read_write> dims: array<u32>;
//[index in values, x, y, z, index in values, x, y, z, ...]

@group(0)
@binding(2)
var<storage, read_write> grad: array<f32>;
//same as values only for gradients

@group(0)
@binding(3)
var<storage, read_write> status: array<f32>;
//status[0] = number of current layer



@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var bias_grad_mat_id = u32(status[0]) * 4u + 2u;
    var output_grad_mat_id = u32(status[0]) * 4u + 3u;

    //indeces in grad array where bias_grad and output_grad start
    var bias_grad_start_index = dims[bias_grad_mat_id * 4u];
    var output_grad_start_index = dims[output_grad_mat_id * 4u];

    //bias_grad = output_grad
    grad[global_id.x + bias_grad_start_index] = grad[global_id.x + output_grad_start_index];
}