use crate::dataset::{CabinSide, HomePlanet, TitanicItem};

use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, ElementConversion, Int, Tensor},
};

#[derive(Clone)]
pub struct TitanicBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TitanicBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TitanicBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<TitanicItem, TitanicBatch<B>> for TitanicBatcher<B> {
    fn batch(&self, items: Vec<TitanicItem>) -> TitanicBatch<B> {
        let mut inputs = Vec::new();

        // pub group_number: u32,
        // pub passenger_number: u8,
        // pub home_planet: HomePlanet,
        // pub cryo_sleep: bool,
        // pub cabin_deck: CabinDeck,
        // pub cabin_number: u32,
        // pub cabin_side: CabinSide,
        // pub desintation: DestinationPlanet,
        // pub age: f32,
        // pub vip: bool,
        // pub room_service: f32,
        // pub food_court: f32,
        // pub shopping_mall: f32,
        // pub spa: f32,
        // pub vr_deck: f32,

        // We should change some of these into one_hot encoding for categories
        for item in items.iter() {
            let total_spending =
                item.room_service + item.food_court + item.shopping_mall + item.spa + item.vr_deck;

            let input_tensor = if total_spending > 0.0 {
                Tensor::<B, 1>::from_floats(
                    [
                        item.group_number as f32,
                        item.passenger_number as f32,
                        item.cabin_number as f32,
                        item.age,
                        // Normalize the spending to a percentage of where the money is spent
                        item.room_service / total_spending,
                        item.food_court / total_spending,
                        item.shopping_mall / total_spending,
                        item.spa / total_spending,
                        item.vr_deck / total_spending,
                        // To call this feature engineering would be kind. Just add up the total spending of an individual
                        total_spending,
                    ],
                    &self.device,
                )
            } else {
                Tensor::<B, 1>::from_floats(
                    [
                        item.group_number as f32,
                        item.passenger_number as f32,
                        item.cabin_number as f32,
                        item.age,
                        // Normalize the spending to a percentage of where the money is spent
                        item.room_service,
                        item.food_court,
                        item.shopping_mall,
                        item.spa,
                        item.vr_deck,
                        // To call this feature engineering would be kind. Just add up the total spending of an individual
                        total_spending,
                    ],
                    &self.device,
                )
            };

            // let input_tensor = Tensor::<B, 1>::from_floats(
            //     [
            //         item.group_number as f32,
            //         item.passenger_number as f32,
            //         item.cabin_number as f32,
            //         item.age,
            //         item.room_service,
            //         item.food_court,
            //         item.shopping_mall,
            //         item.spa,
            //         item.vr_deck,
            //     ],
            //     &self.device,
            // );

            let spent_money = if total_spending == 0.0 {
                Tensor::one_hot(0, 2, &self.device)
            } else {
                Tensor::one_hot(1, 2, &self.device)
            };

            let affluence = if item.age > 0.0 {
                Tensor::<B, 1>::from_floats([total_spending / item.age], &self.device)
            } else {
                Tensor::<B, 1>::from_floats([0.0], &self.device)
            };

            let home_planet: Tensor<B, 1> =
                Tensor::one_hot(item.home_planet as usize, 4, &self.device);
            let cryo_sleep: Tensor<B, 1> =
                Tensor::one_hot(if item.cryo_sleep { 1 } else { 0 }, 2, &self.device);
            let cabin_deck: Tensor<B, 1> =
                Tensor::one_hot(item.cabin_deck as usize, 11, &self.device);
            let cabin_side: Tensor<B, 1> =
                Tensor::one_hot(item.cabin_side as usize, 3, &self.device);
            let destination: Tensor<B, 1> =
                Tensor::one_hot(item.desintation as usize, 4, &self.device);
            let vip: Tensor<B, 1> = Tensor::one_hot(if item.vip { 1 } else { 0 }, 2, &self.device);

            let combination = Tensor::cat(
                vec![
                    input_tensor,
                    affluence,
                    spent_money,
                    home_planet,
                    cryo_sleep,
                    cabin_deck,
                    cabin_side,
                    destination,
                    vip,
                ],
                0,
            );

            inputs.push(combination.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        // let inputs = self.min_max_norm(inputs);

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.transported as u32).elem()], &self.device)
            })
            .collect();

        let targets = Tensor::cat(targets, 0);

        TitanicBatch { inputs, targets }
    }
}
