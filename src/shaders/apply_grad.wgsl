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
//status[1] = learning rate

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    values[global_id.x] = values[global_id.x] - grad[global_id.x] * status[1];
}