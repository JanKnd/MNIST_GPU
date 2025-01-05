use rand::{thread_rng, Rng};

extern crate savefile;
use savefile::prelude::*;
use savefile_derive::Savefile;

#[derive(Savefile)]
pub struct Network {
    pub values: Vec<f32>,
    pub grad: Vec<f32>,
    pub dims: Vec<u32>,
    pub status: Vec<f32>,
}

impl Network {
    pub fn new_xor() -> Network {
        let dims: Vec<u32> = vec![
            0, 2, 1, 1,  2, 3, 2, 1,  8, 3, 1, 1,  11, 3, 1, 1,  14, 3, 1, 1,  17, 3, 3, 1,  26, 3, 1, 1,
            29, 3, 1, 1, 32, 3, 1, 1, 35, 1, 3, 1, 38, 1, 1, 1, 39,1,1,1, 40, 1,1,1,
        ];

        let mut rng = rand::thread_rng();
        //fill values with random values
        let mut values: Vec<f32> = (0..41).map(|_| rng.gen_range(0.0..1.0)).collect();

        /*
        let mut values: Vec<f32> = vec![0.,0., //in
                                        1.,1.,0.5,0.5,
                                        0.,1.,
                                        0.,0., //out
                                        1.,1.,//in
                                        1.,1.,
                                        1.,
                                        1.,
                                        1.]; //out


         */

        let grad: Vec<f32> = vec![0.0; 41];
        let status: Vec<f32> = vec![0., 0.1, 0., 0.,0.,];
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

    pub fn save(&self, filename: &str) {
        save_file(filename, 0, self).unwrap();
    }

    pub fn load(filename: &str) -> Network {
        let network: Network = load_file(filename, 0).unwrap();
        network
    }
}
