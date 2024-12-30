pub struct Data {
    pub values: Vec<f32>,
    pub dims: Vec<u32>,
}

impl Data {
    pub fn new_xor() -> Data {
        let values: Vec<f32> = vec![0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0.];

        let dims: Vec<u32> = vec![
            1, 2, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 5, 1, 1, 1, 6, 2, 1, 1, 8, 1, 1, 1, 9, 2, 1, 1, 11,
            1, 1, 1,
        ];

        Data { values, dims }
    }
}
