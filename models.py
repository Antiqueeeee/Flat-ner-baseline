import torch
from transformers import BertModel

class bertLinear(torch.nn.Module):
    def __init__(self,config):
        super(bertLinear,self).__init__()
        self.model_name = "bertLinear"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name,cache_dir="./cache/", output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size,out_features=self.config.label_num)

    def forward(self,bert_inputs):
        outputs = self.bert(bert_inputs,attention_mask=bert_inputs.ne(0).float())
        sequence_output , cls_output = outputs[0],outputs[1]
        sequence_output = self.dropout(sequence_output)
        outputs = self.linear(sequence_output)
        return outputs
class bertLSTM(torch.nn.Module):
    def __init__(self,config):
        super(bertLSTM, self).__init__()
        self.model_name = "bertLSTM"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name,cache_dir="./cache/", output_hidden_states=True)
        self.lstm = torch.nn.LSTM(input_size=config.bert_hid_size,hidden_size=config.bert_hid_size,batch_first=True)
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.label_num)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self,bert_inputs):
        outputs = self.bert(bert_inputs,attention_mask=bert_inputs.ne(0).float())
        sequence_output , cls_output = outputs[0],outputs[1]
        sequence_output = self.dropout(sequence_output)
        outputs,hidden = self.lstm(sequence_output)
        outputs = self.linear(outputs)
        return outputs
class bertBiLSTM(torch.nn.Module):
    def __init__(self,config):
        super(bertBiLSTM, self).__init__()
        self.model_name = "bertBiLSTM"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name,cache_dir="./cache/", output_hidden_states=True)
        self.lstm = torch.nn.LSTM(
                                   input_size=config.bert_hid_size
                                  ,hidden_size=config.bert_hid_size//2
                                  ,batch_first=True
                                  ,bidirectional=True
                                  )
        self.linear = torch.nn.Linear(in_features=self.config.bert_hid_size, out_features=self.config.label_num)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self,bert_inputs):
        outputs = self.bert(bert_inputs,attention_mask=bert_inputs.ne(0).float())
        sequence_output , cls_output = outputs[0],outputs[1]
        sequence_output = self.dropout(sequence_output)
        outputs,hidden = self.lstm(sequence_output)
        outputs = self.linear(outputs)
        return outputs
class bertCNN(torch.nn.Module):
    def __init__(self,config):
        super(bertCNN, self).__init__()
        self.model_name = "bertCNN"
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_name,cache_dir="./cache/", output_hidden_states=True)
        self.base = torch.nn.Sequential(
            torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(in_channels=self.config.bert_hid_size,out_channels=self.config.conv_hid_size,kernel_size = 1),
            torch.nn.GELU()
        )
        self.conv_ = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=self.config.conv_hid_size
                             ,out_channels=self.config.conv_hid_size
                             ,groups=self.config.conv_hid_size
                             ,kernel_size=3,dilation=d,padding=d) for d in config.dilation]
        )

        self.linear = torch.nn.Linear(in_features=self.config.conv_hid_size * len(config.dilation), out_features=self.config.label_num)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self,bert_inputs):
        sequence_output = self.bert(bert_inputs,bert_inputs.ne(0).byte())[0]
        sequence_output = sequence_output.unsqueeze(1).permute(0,3,1,2)
        sequence_output = self.base(sequence_output)
        conv_outputs = list()
        for conv in self.conv_:
            conv_output = conv(sequence_output)
            conv_output = torch.nn.functional.gelu(conv_output)
            conv_outputs.append(conv_output)
        conv_output = torch.cat(conv_outputs,dim=1).permute(0,2,3,1).squeeze(1)
        conv_output = self.linear(conv_output)
        return conv_output
