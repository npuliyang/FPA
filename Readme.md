

## Environment Required
`pytorch`
`torchattacks`
`torchvision`
## Usage
The first step is to train an victim model.
`python main.py --train victim`

Then to train the adversarial attack model in black box setting.
`python main.py --train attack`

If training the SOTA models, please run:
`python main.py --train sota`

## Citation
