import logging
import torch
import torch.nn as nn

from timeit import default_timer as timer
from train_common import parse_hparams, load_TrainTestDataSet
import optuna
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import optimizer

from transformers import AutoTokenizer
from pytorch_t import (
    Seq2SeqTransformer,
    get_transform, train, evaluate,
    save_model, load_pretrained, load_nmt,
    PAD_IDX, DEVICE,collate_fn,create_mask
)

# from morichan


def get_optimizer(hparams, model):
    # オプティマイザの定義 (Adam)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    return optimizer


def get_optimizer_adamw(hparams, model):
    # オプティマイザの定義 (AdamW)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=hparams.learning_rate,
                                  eps=hparams.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=hparams.warmup_steps,
    #     num_training_steps=t_total
    # )
    return optimizer




#optuna
def get_batch(trial):
  batch_size = trial.suggest_int("batch_size", 32, 1024)
  return batch_size

def get_optimizers(trial, model):

  adam_lr=trial.suggest_loguniform("adam_lr", 2e-5, 2e-4)
  
  optimizer = torch.optim.Adam(
    model.parameters(),
    # transformer.parameters(),
    lr=adam_lr,
    betas=(0.9, 0.98), eps=1e-9
  )
  return optimizer

def object_train(trial,train_iter, model, batch_size, loss_fn): #変更
    optimizer = get_optimizers(trial,model) #変更
    model.train()
    losses = 0

    # 学習データ
    #collate_fn = string_collate(hparams)
    train_dataloader = DataLoader(
        train_iter, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader) ,optimizer #変更



def objective(trial):
  BATCH_SIZE = get_batch(trial)
  print('BATCH_SIZE:',BATCH_SIZE)


  hparams = parse_hparams(setup, Tokenizer=AutoTokenizer)
  _, _, transform = get_transform(
        hparams.tokenizer, hparams.max_length, hparams.target_max_length)
  train_dataset, valid_dataset = load_TrainTestDataSet(
        hparams, transform=transform)

  if hparams.model_name_or_path.endswith('.pt'):
        model = load_pretrained(hparams.model_name_or_path, DEVICE)
  else:
        vocab_size = hparams.vocab_size
        model = Seq2SeqTransformer(hparams.num_encoder_layers, hparams.num_decoder_layers,
                                    hparams.emb_size, hparams.nhead,
                                    vocab_size+4, vocab_size+4, hparams.fnn_hid_dim)

#   print('hparams:wwwwwwwwwwwwwwwwwwww')

  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  train_dataset, valid_dataset = load_TrainTestDataSet(
     hparams, transform=transform)
#   print(train_dataset)
#   print('888888888888888')
#   print(valid_dataset)
  for epoch in range(1, NUM_EPOCHS+1):
      start_time = timer()
    #   print('xxxxxxxxxxxxxxxxx')
    #   print(train_dataset)
    #   print('666666666666')
    #   print(valid_dataset)
    #   optimizer = torch.optim.Adam(
    #     model.parameters(), lr=study.best_params['adam_lr'], betas=(0.9, 0.98), eps=1e-9
    #     )
      train_loss, optimizer = object_train(trial,train_dataset, model, BATCH_SIZE, loss_fn)
    #   train_loss, optimizer = object_train(model, trial, BATCH_SIZE)#modelに変更する　ここ、渡してるものが足りないよ！！！　trial,train_iter, model, batch_size, loss_fn, optimizer
      end_time = timer()
    #   print('3333333333333333333')
    #   print(BATCH_SIZE)
      val_loss = evaluate(valid_dataset, model, BATCH_SIZE, loss_fn)#modelに変更する  val_iter, model, batch_size, loss_fn
    #   val_loss = evaluate(model, BATCH_SIZE)#modelに変更する  val_iter, model, batch_size, loss_fn
      epoch_time=end_time-start_time
    #   print('222222222222222222222222')
      
      #追加
      TRAINLOSS.append(train_loss)
      VALLOSS.append(val_loss)
      EPOCHTIME.append(epoch_time)

      print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
      return val_loss








setup = dict(
    model_name_or_path='megagonlabs/t5-base-japanese-web',
    tokenizer_name_or_path='megagonlabs/t5-base-japanese-web',
    additional_tokens='<nl> <tab> <b> </b> <e0> <e1> <e2> <e3>',
    seed=42,
    encoding='utf_8',
    column=0, target_column=1,
    kfold=5,  # cross validation
    max_length=80,
    target_max_length=80,
    # training
    max_epochs=30,
    num_workers=2,  # os.cpu_count(),
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    # learning_rate=0.0001,
    # adam_epsilon=1e-9,
    # weight_decay=0
    # Transformer
    emb_size=512,  # BERT の次元に揃えれば良いよ
    nhead=8,
    fnn_hid_dim=512,  # 変える
    batch_size=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)



NUM_EPOCHS = 60

#追加
TRAINLOSS=[]
VALLOSS=[]
EPOCHTIME=[]


def _main():
    hparams = parse_hparams(setup, Tokenizer=AutoTokenizer)
    _, _, transform = get_transform(
        hparams.tokenizer, hparams.max_length, hparams.target_max_length)
    train_dataset, valid_dataset = load_TrainTestDataSet(
        hparams, transform=transform)


    # デバイスの指定
    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    if hparams.model_name_or_path.endswith('.pt'):
        model = load_pretrained(hparams.model_name_or_path, DEVICE)
    else:
        vocab_size = hparams.vocab_size
        model = Seq2SeqTransformer(hparams.num_encoder_layers, hparams.num_decoder_layers,
                                   hparams.emb_size, hparams.nhead,
                                   vocab_size+4, vocab_size+4, hparams.fnn_hid_dim)

    # TODO: ?
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('Parameter:', params)
    print('222222222222222222222222222222222222')



    # デバイスの設定
    model = model.to(DEVICE)

    # 損失関数の定義 (クロスエントロピー)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print(train_dataset)
    print('PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP')

    # オプティマイザの定義 (Adam)
    optimizer = get_optimizer(hparams, model)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=8000)
    print('77777777777777777777777777777777777')


    # パラメータの定義
    BATCH_SIZE = study.best_params['batch_size']


    

    # オプティマイザの定義 (Adam)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=study.best_params['adam_lr'], betas=(0.9, 0.98), eps=1e-9
    )

    print('000000000000000000000000000000000000')

    train_list = []
    valid_list = []
    logging.info(f'start training max_epochs={hparams.max_epochs}')
    for epoch in range(1, hparams.max_epochs+1):
        print(epoch)
        start_time = timer()
        # train_loss = train(train_dataset, model,
        #                    hparams.batch_size, loss_fn, optimizer)
        train_loss = train(train_dataset, model,
                           BATCH_SIZE, loss_fn, optimizer)
        train_list.append(train_loss)
        end_time = timer()
        val_loss = evaluate(valid_dataset, model, hparams.batch_size, loss_fn)
        valid_list.append(val_loss)
        logging.info(
            (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))

    save_model(hparams, model,
               f'{hparams.output_dir}/tf_{hparams.project}.pt')

    print('Testing on ', DEVICE)
    train_dataset, valid_dataset = load_TrainTestDataSet(hparams)
    generate = load_nmt(
        f'{hparams.output_dir}/tf_{hparams.project}.pt', AutoTokenizer=AutoTokenizer)
    valid_dataset.test_and_save(
        generate, f'{hparams.output_dir}/result_test.tsv')
    train_dataset.test_and_save(
        generate, f'{hparams.output_dir}/result_train.tsv', max=1000)

# greedy search を使って翻訳結果 (シーケンス) を生成


if __name__ == '__main__':
    _main()
