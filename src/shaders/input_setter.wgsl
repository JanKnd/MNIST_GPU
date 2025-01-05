@group(0)
@binding(0)
var<storage, read_write> values: array<f32>;
//[i,n,p,u,t,w,e,i,g,h,t,s,b,i,a,s,e,s,i,n,p,u,t,...]

@group(0)
@binding(3)
var<storage, read_write> status: array<f32>;
//status[0] = number of current layer
//status[1] = learning rate

@group(1)
@binding(0)
var<storage, read_write> data_values: array<f32>;

@group(1)
@binding(1)
var<storage, read_write> data_dims: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //data_dims[status[2] * 2 * 4] is the index in data_values where the starting index of the example input matrix is stored
    values[global_id.x] = data_values[global_id.x + data_dims[u32(status[2] * 8.)]]; //data_dims[u32(status[2]) * 8]];
}