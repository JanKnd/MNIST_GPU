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

//same as multiplied_value but with transposed b, cols of b are rows of b and vice versa
//used to calculate weight_grad = output_grad * input^T
fn multiplied_value_transposed_b(start_a : u32, start_b: u32, index: u32) -> f32 { //3 0 0
    var row = index / dims[start_b * 4u + 1u]; //row = index / num_cols_of_b // 1/1 = 1
    var col = index % dims[start_b * 4u + 2u]; //col = index % num_cols_of_b // 1%1 = 0
    var sum: f32 = 0.0;
    for(var i : u32 = 0u; i < dims[start_a * 4u + 2u]; i = i + 1u) { //for i in range(num_row_of_a)  //1
        var index_a = dims[start_a * 4u] + row * dims[start_a * 4u + 2u] + i; //index in values array for value of a at row and i //6
        var index_b = dims[start_b * 4u] + i * dims[start_b * 4u + 1u] + col; //0
        sum += grad[index_a] * values[index_b];
    }
    return sum;
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var output_grad_mat_id = u32(status[0]) * 4u + 3u; //3
    var input_mat_id = u32(status[0]) * 4u; //0
    var weight_grad_mat_id = u32(status[0]) * 4u + 1u; //1

    // index in grad array where weight_grad starts
    var weight_grad_start_index = dims[weight_grad_mat_id * 4u]; //2

    //weight_grad = output_grad * input^T
    grad[global_id.x + weight_grad_start_index] = multiplied_value_transposed_b(output_grad_mat_id, input_mat_id, global_id.x);

/*
    if (global_id.x == dims[weight_grad_mat_id * 4u + 1u] * dims[weight_grad_mat_id * 4u + 2u] - 1u && u32(status[0]) > 0u) { //if last element in row
        status[0] = status[0] - 1.0;
    }
*/
}