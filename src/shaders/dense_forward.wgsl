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
var<storage, read_write> status: array<u32>;
//status[0] = number of current layer


//calculate the value of the output matrix at index as a result of multiplying matrix a and b
fn multiplied_value(start_a : u32, start_b: u32, index: u32) -> f32 {
    var row = index / dims[start_b * 4u + 2u]; //row = index / num_cols_of_b (row of i in the output matrix)
    var col = index % dims[start_b * 4u + 2u]; //col = index % num_cols_of_b (col of i in the output matrix)
    var sum: f32 = 0.0;
    for(var i : u32 = 0u; i < dims[start_a * 4u + 2u]; i = i + 1u) {  //for each col in a
        var index_a = dims[start_a * 4u] + row * dims[start_a * 4u + 2u] + i; //index in values array for value of a at row and i
        var index_b = dims[start_b * 4u] + i * dims[start_b * 4u + 2u] + col; //index in values array for value of b at i and col
        sum += values[index_a] * values[index_b];
    }
    return sum;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var input_mat_id = status[0] * 4u;
    var weight_mat_id = status[0] * 4u + 1u;
    var bias_mat_id = status[0] * 4u + 2u;
    var output_mat_id = status[0] * 4u + 3u;

    //indeces in values array where output and bias start
    var bias_start_index = dims[bias_mat_id * 4u];
    var output_start_index = dims[output_mat_id * 4u];

    //output = weight * input + bias
    values[global_id.x + output_start_index] = multiplied_value(weight_mat_id, input_mat_id, global_id.x) + values[global_id.x + bias_start_index];


    //if the last element of the output matrix is calculated, increment the current layer number
    if (global_id.x == dims[output_mat_id * 4u + 1u] - 1u) {
        status[0] = status[0] + 1u;
    }
}

