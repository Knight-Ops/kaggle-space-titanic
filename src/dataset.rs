use burn::data::dataset::transform::Mapper;
use burn::data::dataset::transform::MapperDataset;
use burn::data::dataset::Dataset;
use burn::data::dataset::InMemDataset;
use serde::{Deserialize, Serialize};

// PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
// HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
// CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
// Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
// Destination - The planet the passenger will be debarking to.
// Age - The age of the passenger.
// VIP - Whether the passenger has paid for special VIP service during the voyage.
// RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
// Name - The first and last names of the passenger.
// Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum HomePlanet {
    Earth,
    Mars,
    Europa,
    None,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum DestinationPlanet {
    Trappist,
    Pso,
    Cancri,
    None,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum CabinDeck {
    A = 0,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum CabinSide {
    Port = 0,
    Starboard = 1,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TitanicItem {
    pub group_number: u32,
    pub passenger_number: u32,
    pub home_planet: u32,
    pub cryo_sleep: bool,
    pub cabin_deck: u32,
    pub cabin_number: u32,
    pub cabin_side: u32,
    pub desintation: u32,
    pub age: f32,
    pub vip: bool,
    pub room_service: f32,
    pub food_court: f32,
    pub shopping_mall: f32,
    pub spa: f32,
    pub vr_deck: f32,
    // name: String,
    pub transported: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TitanicItemRaw {
    #[serde(alias = "PassengerId")]
    passenger_id: String,
    #[serde(alias = "HomePlanet")]
    home_planet: String,
    #[serde(alias = "CryoSleep")]
    cryo_sleep: Option<String>,
    #[serde(alias = "Cabin")]
    cabin: Option<String>,
    #[serde(alias = "Destination")]
    desination: String,
    #[serde(alias = "Age")]
    age: Option<f32>,
    #[serde(alias = "VIP")]
    vip: Option<String>,
    #[serde(alias = "RoomService")]
    room_service: Option<f32>,
    #[serde(alias = "FoodCourt")]
    food_court: Option<f32>,
    #[serde(alias = "ShoppingMall")]
    shopping_mall: Option<f32>,
    #[serde(alias = "Spa")]
    spa: Option<f32>,
    #[serde(alias = "VRDeck")]
    vr_deck: Option<f32>,
    #[serde(alias = "Name")]
    name: Option<String>,
    #[serde(alias = "Transported")]
    transported: String,
}

// fn default_if_empty<'de, D, T>(de: D) -> Result<T, D::Error>
// where
//     D: serde::Deserializer<'de>,
//     T: serde::Deserialize<'de> + Default,
// {
//     use serde::Deserialize;
//     Option::<T>::deserialize(de).map(|x| x.unwrap_or_else(|| T::default()))
// }

struct RawToItem;

impl Mapper<TitanicItemRaw, TitanicItem> for RawToItem {
    fn map(&self, item: &TitanicItemRaw) -> TitanicItem {
        let cabin = item
            .cabin
            .as_ref()
            .unwrap()
            .split('/')
            .collect::<Vec<&str>>();

        let cabin_deck = match cabin[0] {
            "A" => 0,
            "B" => 1,
            "C" => 2,
            "D" => 3,
            "E" => 4,
            "F" => 5,
            "G" => 6,
            "T" => 7,
            _ => panic!("Invalid cabin deck: {}", format!("{}", cabin[0])),
        };
        let cabin_side = match cabin[2] {
            "P" => 0,
            "S" => 1,
            _ => panic!("Invalid cabin side : {}", format!("{}", cabin[2])),
        };

        let home_planet = match item.home_planet.as_str() {
            "Earth" => 1,
            "Mars" => 2,
            "Europa" => 3,
            "" => 0,
            _ => panic!("Invalid home planet"),
        };

        let cryo_sleep = match item.cryo_sleep.clone().unwrap().as_str() {
            "True" => true,
            "False" => false,
            _ => panic!("Invalid cryo sleep"),
        };

        let destination_planet = match item.desination.as_str() {
            "TRAPPIST-1e" => 1,
            "PSO J318.5-22" => 2,
            "55 Cancri e" => 3,
            "" => 0,
            _ => panic!("Invalid destination planet"),
        };

        let vip = match item.vip.clone().unwrap().as_str() {
            "True" => true,
            "False" => false,
            _ => panic!("Invalid vip"),
        };

        let transported = match item.transported.as_str() {
            "True" => true,
            "False" => false,
            _ => panic!("Invalid transported"),
        };

        TitanicItem {
            group_number: item.passenger_id[0..4].parse().unwrap(),
            passenger_number: item.passenger_id[5..7].parse().unwrap(),
            home_planet: home_planet,
            cryo_sleep: cryo_sleep,
            cabin_deck,
            cabin_number: cabin[1].parse().unwrap(),
            cabin_side,
            desintation: destination_planet,
            age: item.age.unwrap(),
            vip: vip,
            room_service: item.room_service.unwrap(),
            food_court: item.food_court.unwrap(),
            shopping_mall: item.shopping_mall.unwrap(),
            spa: item.spa.unwrap(),
            vr_deck: item.vr_deck.unwrap(),
            // name: item.name.clone(),
            transported: transported,
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<TitanicItemRaw>, RawToItem, TitanicItemRaw>;

pub struct TitanicDataset {
    dataset: MappedDataset,
}

impl Dataset<TitanicItem> for TitanicDataset {
    fn get(&self, index: usize) -> Option<TitanicItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl TitanicDataset {
    pub fn train() -> Self {
        let dataset = InMemDataset::new(
            csv::ReaderBuilder::new()
                .delimiter(b',')
                .terminator(csv::Terminator::CRLF)
                .from_path("data/train.csv")
                .unwrap()
                .into_deserialize::<TitanicItemRaw>()
                .map(|res| res.unwrap())
                // Filter out garbage data here, for the time being, we are just saying any entry that doesn't have an age is garbage
                .filter(|item| {
                    item.age.is_some()
                        && item.food_court.is_some()
                        && item.room_service.is_some()
                        && item.shopping_mall.is_some()
                        && item.spa.is_some()
                        && item.vr_deck.is_some()
                        && item.cryo_sleep.is_some()
                        && item.vip.is_some()
                        && item.cabin.is_some()
                })
                .collect(),
        );

        let dataset_length = dataset.len();

        let train_dataset = InMemDataset::new(
            dataset
                .iter()
                .take((dataset_length as f32 * (9. / 10.)).round() as usize)
                .collect(),
        );

        let dataset = MapperDataset::new(train_dataset, RawToItem);
        Self { dataset }
    }

    // We don't actually have a test dataset for validation, so we need to make our own, we will just keep 10% of the training set to validate
    pub fn test() -> Self {
        let dataset = InMemDataset::new(
            csv::ReaderBuilder::new()
                .delimiter(b',')
                .terminator(csv::Terminator::CRLF)
                .from_path("data/train.csv")
                .unwrap()
                .into_deserialize::<TitanicItemRaw>()
                .map(|res| res.unwrap())
                // Filter out garbage data here, for the time being, we are just saying any entry that doesn't have an age is garbage
                .filter(|item| {
                    item.age.is_some()
                        && item.food_court.is_some()
                        && item.room_service.is_some()
                        && item.shopping_mall.is_some()
                        && item.spa.is_some()
                        && item.vr_deck.is_some()
                        && item.cryo_sleep.is_some()
                        && item.vip.is_some()
                        && item.cabin.is_some()
                })
                .collect(),
        );

        let dataset_length = dataset.len();

        let test_dataset = InMemDataset::new(
            dataset
                .iter()
                .skip((dataset_length as f32 * (9. / 10.)).round() as usize)
                .collect(),
        );

        let dataset = MapperDataset::new(test_dataset, RawToItem);
        Self { dataset }
    }
}
