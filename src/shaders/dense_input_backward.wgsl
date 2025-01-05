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

//same as multiplied_value in dense_forward.wgsl but with transposed a, rows of a are cols of a and vice versa
//used to calculate input_grad = weight^T * output_grad
//accesses grad array
fn multiplied_value_transposed_a(mat_id_a : u32, mat_id_b: u32, index: u32) -> f32 {
    var row = index / dims[mat_id_b * 4u + 2u];
    var col = index % dims[mat_id_b * 4u + 2u];
    var sum: f32 = 0.0;
    for(var i : u32 = 0u; i < dims[mat_id_a * 4u + 1u]; i = i + 1u) {
        var index_a = dims[mat_id_a * 4u] + row * dims[mat_id_a * 4u + 1u] + i;
        var index_b = dims[mat_id_b * 4u] + i * dims[mat_id_b * 4u + 2u] + col;
        sum += values[index_a] * grad[index_b];
    }
    return sum;
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var output_grad_mat_id = u32(status[0]) * 4u + 3; //0
    var weight_mat_id = u32(status[0]) * 4u + 1u; //1
    var input_grad_mat_id = u32(status[0]) * 4u; //3

    //index in grad array where input_grad and output_grad start
    var input_grad_start_index = dims[input_grad_mat_id * 4u];
    var output_grad_start_index = dims[output_grad_mat_id * 4u];

    //input_grad = weight^T * output_grad
    grad[global_id.x + input_grad_start_index] = multiplied_value_transposed_a(weight_mat_id, output_grad_mat_id, global_id.x);
    //grad[global_id.x + input_grad_start_index] = 111.;

    //if the ast element of the output matrix is calculated, decrement the current layer number

    if (global_id.x == dims[output_grad_mat_id * 4u + 1u] - 1u && u32(status[0]) > 0u) {
        status[0] = status[0] - 1.;
    }



}