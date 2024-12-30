@group(0)
@binding(0)
var<storage, read_write> values: array<f32>;
//[i,n,p,u,t,w,e,i,g,h,t,s,b,i,a,s,e,s,o,u,t,p,u,t,i,n,p,u,t,...]

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


/*
//..._mat_id = index in dims array, where index for values is stored
fn multiplied_value(a_mat_id : u32, b_mat_id: u32, index_in_result: u32) -> f32 {
    var row = index_in_result / dims[b_mat_id + 2u]; //row = index / num_cols_of_b, row is row in result matrix
    var col = index_in_result % dims[b_mat_id + 2u]; //col = index % num_cols_of_b, col is col in result matrix
    var result: f32 = 0.0;
    for (var i = 0u; i < dims[a_mat_id + 2u]; i = i + 1u) { //for i in range(num_cols_of_a)
                //result += a[res_row][i] * b[i][res_col]
                result = result + values[dims[a_mat_id] + res_row * dims[a_mat_id + 2u] + i] * values[dims[b_mat_id] + i * dims[b_mat_id + 2u] + res_col];
    }
    return result;
}

fn multiplied_value_transposed_a() -> f32 {
    var row = index_in_result / dims[b_mat_id + 2u]; //row = index / num_cols_of_b, row is row in result matrix
    var col = index_in_result % dims[b_mat_id + 2u]; //col = index % num_cols_of_b, col is col in result matrix
    var result: f32 = 0.0;
    for (var i = 0u; i < dims[a_mat_id + 1u]; i = i + 1u) { //for i in range(num_cols_of_a)
                //result += a[res_row][i] * b[i][res_col]
                result = result + values[dims[a_mat_id] + res_row * dims[a_mat_id + 1u] + i] * values[dims[b_mat_id] + i * dims[b_mat_id + 2u] + res_col];
    }
    return result;
}

fn multiplied_value_transposed_b() -> f32 {
    var row = index_in_result / dims[b_mat_id + 2u]; //row = index / num_cols_of_b, row is row in result matrix
    var col = index_in_result % dims[b_mat_id + 2u]; //col = index % num_cols_of_b, col is col in result matrix
    var result: f32 = 0.0;
    for (var i = 0u; i < dims[a_mat_id + 2u]; i = i + 1u) { //for i in range(num_cols_of_a)
                //result += a[res_row][i] * b[i][res_col]
                result = result + values[dims[a_mat_id] + res_row * dims[a_mat_id + 2u] + i] * values[dims[b_mat_id] + i * dims[b_mat_id + 2u] + res_col];
    }
    return result;
} */


fn multiplied_value(start_a : u32, start_b: u32, index: u32) -> f32 {
    var row = index / dims[start_b * 4u + 2u]; //row = index / num_cols_of_b
    var col = index % dims[start_b * 4u + 2u]; //col = index % num_cols_of_b
    var sum: f32 = 0.0;
    for(var i : u32 = 0u; i < dims[start_a * 4u + 2u]; i = i + 1u) {  //for each col in a
        var index_a = dims[start_a * 4u] + row * dims[start_a * 4u + 2u] + i; //index in values array for value of a at row and i
        var index_b = dims[start_b * 4u] + i * dims[start_b * 4u + 2u] + col; //index in values array for value of b at i and col
        sum += values[index_a] * values[index_b];
    }
    return sum;
}

//same as multiplied value but with transposed a, rows of a are cols of a and vice versa
//used to calculate input_grad = weight^T * output_grad
//accesses grad array
fn multiplied_value_transposed_a(start_a : u32, start_b: u32, index: u32) -> f32 {
    var row = index / dims[start_b * 4u + 2u];
    var col = index % dims[start_b * 4u + 2u];
    var sum: f32 = 0.0;
    for(var i : u32 = 0u; i < dims[start_a * 4u + 1u]; i = i + 1u) {
        var index_a = dims[start_a * 4u] + row * dims[start_a * 4u + 1u] + i;
        var index_b = dims[start_b * 4u] + i * dims[start_b * 4u + 2u] + col;
        sum += values[index_a] * grad[index_b];
    }
    return sum;
}

//same as multiplied_value but with transposed b, cols of b are rows of b and vice versa
//used to calculate weight_grad = output_grad * input^T
fn multiplied_value_transposed_b(start_a : u32, start_b: u32, index: u32) -> f32 {
    var row = index / dims[start_b * 4u + 2u]; //row = index / num_cols_of_b
    var col = index % dims[start_b * 4u + 2u]; //col = index % num_cols_of_b
    var sum: f32 = 0.0;
    for(var i : u32 = 0u; i < dims[start_a * 4u + 2u]; i = i + 1u) { //for i in range(num_row_of_a)
        var index_a = dims[start_a * 4u] + row * dims[start_a * 4u + 2u] + i; //index in values array for value of a at row and i
        var index_b = dims[start_b * 4u] + i * dims[start_b * 4u + 1u] + col;
        sum += grad[index_a] * values[index_b];
    }
    return sum;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    switch(status[1]){
        //forward pass
        case 0u: {
            var input_mat_id = status[0] * 4u;
            var weight_mat_id = status[0] * 4u + 1u;
            var bias_mat_id = status[0] * 4u + 2u;
            var output_mat_id = status[0] * 4u + 3u;
            //output = weight * input + bias
            values[global_id.x + dimsoutput_mat_id * 4u] = multiplied_value(weight_mat_id, input_mat_id, global_id.x) + values[global_id.x + bias_mat_id * 4u];
        }

        //input_grad
        case 1u: {
            var output_grad_mat_id = status[0] * 4u;
            var weight_mat_id = status[0] * 4u + 1u;
            var input_grad_mat_id = status[0] * 4u + 3u;
            //input_grad = weight^T * output_grad
            grad[global_id.x + input_grad_mat_id * 4u] = multiplied_value_transposed_a(weights_mat_id, output_grad_id, global_id.x);
        }
        //weight_grad
        case 2u: {
            var output_grad_mat_id = status[0] * 4u;
            var input_mat_id = status[0] * 4u + 1u;
            var weight_grad_mat_id = status[0] * 4u + 2u;
            //weight_grad = output_grad * input^T
            grad[global_id.x + weight_grad_mat_id * 4u] = multiplied_value_transposed_b(output_grad_mat_id, input_grad_mat_id, global_id.x);
        }
        //bias_grad
        case 3u: {
            //TODO
        }
}