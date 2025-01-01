use rand::{thread_rng, Rng};
pub struct Network {
    pub values: Vec<f32>,
    pub grad: Vec<f32>,
    pub dims: Vec<u32>,
    pub status: Vec<f32>,
}

impl Network {
    pub fn new_xor() -> Network {
        let dims: Vec<u32> = vec![
            0, 2, 1, 1, 2, 2, 2, 1, 6, 2, 1, 1, 8, 2, 1, 1, 10, 2, 1, 1, 12, 1, 2, 1, 14, 1, 1, 1,
            15, 1, 1, 1,
        ];

        let mut rng = rand::thread_rng();
        //fill values with random values
        let mut values: Vec<f32> = (0..16).map(|_| rng.gen_range(0.0..1.0)).collect();
        let grad: Vec<f32> = vec![0.0; 16];
        let status: Vec<f32> = vec![0., 0.001, 0., 0.,0.,];
/*
        status[0] = current layers
        status[1] = learning rate
        status[2] = current training set
        status[3] = mse added up
        status[4] = mse count
        status[5] = input size
        status[6] = output size
        status[7] = output_mat_start
            */
        Network {
            values,
            grad,
            dims,
            status,
        }
    }
}
