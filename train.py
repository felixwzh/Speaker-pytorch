from persona import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', type=str, default='data/testing',
					help='the folder that contains your dataset and vocabulary file')
parser.add_argument('--train_file', type=str, default='train.txt')
parser.add_argument('--dev_file', type=str, default='valid.txt')
parser.add_argument('--dictPath', type=str, default='vocabulary')
parser.add_argument('--save_folder', type=str, default='save/testing')
parser.add_argument('--save_prefix', type=str, default='model')
parser.add_argument('--save_params', type=str, default='params')
parser.add_argument('--output_file', type=str, default='log')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--UNK',type=int,default=0,
					help='the index of UNK. UNK+special_word=3.')
parser.add_argument('--special_word', type=int, default=3,
					help='default special words include: padding, EOS, EOT.')

parser.add_argument('--fine_tuning', action='store_true')
parser.add_argument('--fine_tunine_model', type=str, default='model')

parser.add_argument('--PersonaNum', type=int, default=4167)
parser.add_argument('--SpeakerMode', action='store_true')
parser.add_argument('--AddresseeMode', action='store_true')

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--source_max_length", type=int, default=128)
parser.add_argument("--target_max_length", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=10)

parser.add_argument("--dimension", type=int, default=512)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--init_weight", type=float, default=0.1)

parser.add_argument("--alpha", type=int, default=1)
parser.add_argument("--start_halve", type=int, default=6)
parser.add_argument("--thres", type=int, default=5)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--save_steps", type=int, default=2)
parser.add_argument("--eval_steps", type=int, default=5000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--train_size", type=int, default=960114)



parser.add_argument('--PersonaDim', type=int, default=40)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--PersonaEmbFiles', type=str, default='None')
parser.add_argument("--best_ppl_threshold", type=float, default=60.0)


parser.add_argument('--get_persona_emb', action='store_true')
parser.add_argument('--output_persona_emb_in_training', action='store_true')
parser.add_argument('--persona_emb_output_file', type=str, default='None')



args = parser.parse_args()
print(args)
print()

if __name__ == '__main__':
	model = persona(args)
	model.train()