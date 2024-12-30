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

//activation function to be applied to every element of the output matrix of the prev layer
fn relu_derivative(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

fn sigmoid_derivative(x: f32) -> f32 {
    return x * (1.0 - x);
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //grad_x = grad_Y * grad_activation_fn

    //input_grad is output matrix of prev layer as status[0] is incremented while dense_forward.wgsl is executed
    var input_grad_mat_id = u32(status[0]) * 4u - 1u;
    //output_grad is input matrix of current layer
    var output_grad_mat_id = u32(status[0]) * 4u;

    //apply activation function to every element of the output matrix of the prev layer
    grad[global_id.x + dims[input_grad_mat_id * 4u]] = grad[global_id.x + dims[output_grad_mat_id * 4u]] * sigmoid_derivative(values[global_id.x + dims[input_grad_mat_id * 4u]]);

    //switch
}