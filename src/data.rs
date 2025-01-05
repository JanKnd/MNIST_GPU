use rand::Rng;

#[derive(Debug)]
#[derive(Clone)]
pub struct Data {
    pub values: Vec<f32>,
    pub dims: Vec<u32>,
}


impl Data {
    pub fn new_xor() -> Data {

        let values: Vec<f32> = vec![0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0.];

        let dims: Vec<u32> = vec![
            0, 2, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 5, 1, 1, 1, 6, 2, 1, 1, 8, 1, 1, 1, 9, 2, 1, 1, 11,
            1, 1, 1,
        ];


/*
        let values: Vec<f32> = vec![0., 0., 0.,];

        let dims: Vec<u32> = vec![
            0, 2, 1, 1, 2, 1, 1, 1,
        ];
        */
        Data { values, dims }
    }

    pub fn new_xor_len(example_count: usize) -> Data {
        let mut rng = rand::thread_rng();
        let a = vec![0., 0., 0.,];
        let b = vec![0., 1., 1.,];
        let c = vec![1., 0., 1.,];
        let d = vec![1., 1., 0.,];
        let mut values = vec![];
        let mut dims: Vec<u32> = vec![];
        for i in 0..example_count {
            let choice = rng.gen_range(0..4);
            match choice {
                0 => values.extend_from_slice(&a),
                1 => values.extend_from_slice(&b),
                2 => values.extend_from_slice(&c),
                3 => values.extend_from_slice(&d),
                _ => panic!("unexpected choice"),
            }
            let mut d = vec![(i * 3) as u32, 2u32, 1u32, 1u32, (i * 3 + 2) as u32, 1u32, 1u32, 1u32];
            dims.append(&mut d);
        }
        Data { values, dims }

    }

}
