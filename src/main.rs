use rand::prelude::*;

use bevy::{prelude::*, sprite, window::PresentMode};

use dfdx::losses::mse_loss;
use dfdx::nn::{Linear, Module, ReLU, ResetParams};
use dfdx::optim::{Adam, Optimizer};
use dfdx::tensor::{HasArrayData, Tensor1D, Tensor2D, TensorCreator};
use dfdx::tensor_ops::Select1;

const NEXT_STATE_DISCOUNT: f32 = 0.9;
const BATCH_SIZE: usize = 64;
const EPSILON_DECAY: f32 = 0.0002;

// Cart pole variables from OpenAI
const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_POLE * MASS_CART;
const LENGTH: f32 = 0.5;
const POLEMASS_LENGTH: f32 = MASS_POLE * LENGTH;
const FORCE_MAG: f32 = 10.;
const TAU: f32 = 0.002;
const THETA_THRESHOLD_RADIANS: f32 = 12. * 2. * std::f32::consts::PI / 360.;
const X_THRESHOLD: f32 = 2.4;

const ARENA_WIDTH: f32 = 2. * X_THRESHOLD;
const ARENA_HEIGHT: f32 = ARENA_WIDTH * (9. / 16.);

type Mlp = (
    Linear<4, 64>,
    (Linear<64, 64>, ReLU),
    (Linear<64, 32>, ReLU),
    Linear<32, 2>,
);

type Transition = ([f32; 4], i32, i32, Option<[f32; 4]>);

#[derive(Debug, Default)]
struct Model {
    model: Mlp,
    target: Mlp,
    optimizer: Adam<Mlp>,
    steps_since_last_merge: i32,
    survived_steps: i32,
    episode: i32,
    epsilon: f32,
    experience: Vec<Transition>,
}

impl Model {
    pub fn default() -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let mut mlp = Mlp::default();
        let mut target = Mlp::default();
        mlp.reset_params(&mut rng);
        target.reset_params(&mut rng);
        Self {
            model: mlp,
            target,
            optimizer: Adam::default(),
            steps_since_last_merge: 0,
            survived_steps: 0,
            episode: 0,
            epsilon: 1.,
            experience: Vec::new(),
        }
    }

    pub fn push_experience(&mut self, transition: Transition) {
        self.experience.push(transition);
        if self.experience.len() > 10000 {
            self.experience = self.experience[5000..].to_vec();
        }
    }

    pub fn train(&mut self) {
        // Select the experience batch
        let mut rng = rand::thread_rng();
        let distribution = rand::distributions::Uniform::from(0..self.experience.len());
        let experience: Vec<Transition> = (0..BATCH_SIZE)
            .map(|_index| self.experience[distribution.sample(&mut rng)])
            .collect();

        // Get the models expected rewards
        let observations: Vec<_> = experience.iter().map(|x| x.0.to_owned()).collect();
        let observations: [[f32; 4]; BATCH_SIZE] = observations.try_into().unwrap();
        let observations: Tensor2D<BATCH_SIZE, 4> = TensorCreator::new(observations);
        let predictions = self.model.forward(observations.trace());
        let actions_indices: Vec<_> = experience.iter().map(|x| x.1 as usize).collect();
        let actions_indices: [usize; BATCH_SIZE] = actions_indices.try_into().unwrap();
        let predictions: Tensor1D<BATCH_SIZE, dfdx::prelude::OwnedTape> =
            predictions.select(&actions_indices);

        // Get the targets expected rewards for the next_observation
        // This could be optimized but I can't think of a easy way to do it without making this
        // code much more gross, and since we are already far faster than we need to be, this is
        // fine BUT when not rendering the window, this is the bottleneck in the program
        let mut target_predictions: [f32; BATCH_SIZE] = [0.; BATCH_SIZE];
        for (i, x) in experience.iter().enumerate() {
            let target_prediction = match x.3 {
                Some(next_observation) => {
                    let next_observation: Tensor1D<4> = TensorCreator::new(next_observation);
                    let target_prediction = self.target.forward(next_observation);
                    let target_prediction =
                        target_prediction.data()[0].max(target_prediction.data()[1]);
                    target_prediction * NEXT_STATE_DISCOUNT + experience[i].2 as f32
                }
                None => experience[i].2 as f32,
            };
            target_predictions[i] = target_prediction;
        }
        let target_predictions: Tensor1D<BATCH_SIZE> = TensorCreator::new(target_predictions);

        // Get the loss and train the model
        let loss = mse_loss(predictions, &target_predictions);
        self.optimizer
            .update(&mut self.model, loss.backward())
            .expect("Oops, we messed up");
    }
}

#[derive(Component)]
struct Cart;

#[derive(Component)]
struct Pole;

#[derive(Component)]
struct Size {
    width: f32,
    height: f32,
}

#[derive(Component)]
struct Velocity(f32);

impl Velocity {
    pub fn default() -> Self {
        Self(0.)
    }
}

fn main() {
    App::new()
        .insert_resource(WindowDescriptor {
            title: "Cart Pole".to_string(),
            present_mode: PresentMode::AutoVsync,
            ..default()
        })
        .add_plugins(DefaultPlugins)
        .add_startup_system(add_camera)
        .add_system(size_scaling)
        // Minimal plugins is what we want when we are not rendering the window and are just
        // testing the training
        // .add_plugins(MinimalPlugins)
        .add_startup_system(add_cart_pole)
        .add_startup_system(add_model.exclusive_system())
        .add_system(step)
        .run();
}

fn size_scaling(windows: Res<Windows>, mut q: Query<(&Size, &mut Transform)>) {
    let window = windows.get_primary().unwrap();
    for (sprite_size, mut transform) in q.iter_mut() {
        transform.scale = Vec3::new(
            sprite_size.width / ARENA_WIDTH as f32 * window.width() as f32,
            sprite_size.height / ARENA_HEIGHT as f32 * window.height() as f32,
            1.0,
        );
    }
}

fn step(
    mut q_text: Query<&mut Text>,
    mut q_cart: Query<(&mut Transform, &mut Velocity), (With<Cart>, Without<Pole>)>,
    mut q_pole: Query<(&mut Transform, &mut Velocity), (With<Pole>, Without<Cart>)>,
    mut model: NonSendMut<Model>,
) {
    let (mut cart_transform, mut cart_velocity) = q_cart
        .get_single_mut()
        .expect("Could not get the cart information");
    let (mut pole_transform, mut pole_velocity) = q_pole
        .get_single_mut()
        .expect("Could not get the pole information");
    let mut text = q_text
        .get_single_mut()
        .expect("Could not get the text with the episode info");

    let observation = [
        cart_transform.translation.x,
        cart_velocity.0,
        pole_transform.rotation.z,
        pole_velocity.0,
    ];

    let action = match model.epsilon > rand::random::<f32>() {
        true => match rand::random::<bool>() {
            true => 0,
            false => 1,
        },
        false => {
            let tensor_observation: Tensor1D<4> = TensorCreator::new(observation);
            let prediction = model.model.forward(tensor_observation);
            match prediction.data()[0] > prediction.data()[1] {
                true => 0,
                false => 1,
            }
        }
    };
    model.epsilon = (model.epsilon - EPSILON_DECAY).max(0.05);

    // These calculations are directly from openai https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    let force = match action {
        1 => FORCE_MAG * -1.,
        _ => FORCE_MAG,
    };
    let costheta = pole_transform.rotation.z.cos();
    let sintheta = pole_transform.rotation.z.sin();
    let temp =
        (force + POLEMASS_LENGTH * pole_transform.rotation.z.powi(2) * sintheta) / TOTAL_MASS;
    let thetaacc = (GRAVITY * sintheta - costheta * temp)
        / (LENGTH * (4.0 / 3.0 - MASS_POLE * (costheta * costheta) / TOTAL_MASS));
    let xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

    // Apply above calculations
    cart_transform.translation.x += TAU * cart_velocity.0 * cart_transform.scale.x;
    cart_velocity.0 += TAU * xacc;
    pole_transform.rotation.z += TAU * pole_velocity.0;
    pole_velocity.0 += TAU * thetaacc;
    // Match the pole x to the cart x
    pole_transform.translation.x = cart_transform.translation.x;

    // Check if the episode is over
    if pole_transform.rotation.z > THETA_THRESHOLD_RADIANS
        || pole_transform.rotation.z < -1. * THETA_THRESHOLD_RADIANS
        || (cart_transform.translation.x / cart_transform.scale.x) > X_THRESHOLD
        || (cart_transform.translation.x / cart_transform.scale.x) < -1. * X_THRESHOLD
        || model.survived_steps > 499
    {
        println!(
            "RESETTING Episode: {}  SURVIVED: {}",
            model.episode, model.survived_steps,
        );

        // Reset cart and pole variables just like openai does
        let mut rng = rand::thread_rng();
        cart_velocity.0 = rng.gen_range(-0.05..0.05);
        pole_velocity.0 = rng.gen_range(-0.05..0.05);
        cart_transform.translation.x = rng.gen_range(-0.05..0.05);
        pole_transform.translation.x = cart_transform.translation.x;
        pole_transform.rotation.z = rng.gen_range(-0.05..0.05);

        // Update the latest episode and survived text
        text.sections[0].value = format!(
            "Episode: {} - Survided: {}",
            model.episode, model.survived_steps
        );

        // Reset the survived_steps, increment episode count, and push_experience
        model.survived_steps = 0;
        model.episode += 1;
        model.push_experience((observation, action, 0, None));
    } else {
        model.survived_steps += 1;
        let next_observation = [
            cart_transform.translation.x,
            cart_velocity.0,
            pole_transform.rotation.z,
            pole_velocity.0,
        ];
        model.push_experience((observation, action, 1, Some(next_observation)));
    }

    // Train if we have the necessary experience
    if model.experience.len() > BATCH_SIZE {
        model.train();
    }

    // Merge the target model after a certain number of steps
    if model.steps_since_last_merge > 10 {
        model.target = model.model.clone();
        model.steps_since_last_merge = 0;
    } else {
        model.steps_since_last_merge += 1;
    }
}

fn add_cart_pole(mut commands: Commands, asset_server: Res<AssetServer>) {
    let cart_handle = asset_server.load("cart.png");
    let pole_handle = asset_server.load("pole.png");
    commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                custom_size: Some(Vec2::new(1., 1.)),
                ..default()
            },
            texture: cart_handle,
            transform: Transform {
                translation: Vec3::new(0., 0., 0.),
                scale: Vec3::new(1., 1., 1.),
                ..default()
            },
            ..default()
        })
        .insert(Cart)
        .insert(Velocity::default())
        .insert(Size {
            width: 0.6,
            height: 0.3,
        });
    commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                anchor: sprite::Anchor::BottomCenter,
                custom_size: Some(Vec2::new(1., 1.)),
                ..default()
            },
            texture: pole_handle,
            transform: Transform {
                translation: Vec3::new(0., 0., 1.),
                scale: Vec3::new(1., 1., 1.),
                ..default()
            },
            ..default()
        })
        .insert(Pole)
        .insert(Velocity::default())
        .insert(Size {
            width: 0.1,
            height: 1.,
        });

    commands.spawn_bundle(
        TextBundle::from_section(
            "",
            TextStyle {
                font: asset_server.load("fonts/FiraSans-Bold.otf"),
                font_size: 20.0,
                color: Color::WHITE,
            },
        )
        .with_text_alignment(TextAlignment::CENTER)
        .with_style(Style {
            align_self: AlignSelf::FlexEnd,
            position_type: PositionType::Absolute,
            position: UiRect {
                bottom: Val::Px(5.0),
                right: Val::Px(15.0),
                ..default()
            },
            ..default()
        }),
    );
}

fn add_camera(mut commands: Commands) {
    commands.spawn_bundle(Camera2dBundle::default());
}

fn add_model(world: &mut World) {
    world.insert_non_send_resource(Model::default());
}
