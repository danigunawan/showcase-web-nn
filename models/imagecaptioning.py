import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.utils import weight_norm

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, pad, dropout = 0.2):
        super(ResidualBlock1D, self).__init__()
        pad0 = torch.nn.ZeroPad2d((pad, 0))
        conv0 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation))
        relu0 = nn.ReLU(inplace=True)
        dropout0 = nn.Dropout(dropout)

        pad1 = torch.nn.ZeroPad2d((pad, 0))
        conv1 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, dilation=dilation))
        relu1 = nn.ReLU(inplace=True)
        dropout1 = nn.Dropout(dropout)

        self.layers = nn.Sequential(pad0, conv0, relu0, dropout0, pad1, conv1, relu1, dropout1)

        self.channel_scaling = None
        if in_channels != out_channels:
            self.channel_scaling = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1))


    def forward(self, x):
        res = x
        out = self.layers(x)
        if self.channel_scaling != None:
            res = self.channel_scaling(x)
        return res + out

    
class TimeSeriesCNN(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size):
        super(TimeSeriesCNN, self).__init__()
        blocks = []
        for i in range(n_layers):
            dil = 1
            #dil = 1 + 1*i
            #dil = int(1+i*0.5)
            n_prev_channels = hidden_size if i != 0 else input_size
            n_channels = hidden_size if i != n_layers-1 else output_size
            blocks += [ResidualBlock1D(n_prev_channels, n_channels, 3, dil, pad=dil*(3-1), dropout=0.2)]
        self.residual_blocks = nn.Sequential(*blocks)


    def forward(self, x):
        out = self.residual_blocks(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules_features = list(resnet.children())[:9]
        self.resnet_features = nn.Sequential(*modules_features)
        

    def forward(self, x):
        x = self.resnet_features(x)
        return x.view(-1, 512, 1)


class ImageAnnotator(nn.Module):
    def __init__(self, n_layers, hidden_size, lang):
        super(ImageAnnotator, self).__init__()
        self.lang = lang
        self.feature_extractor = FeatureExtractor()
        self.embeddings = nn.Embedding(lang.vocab_size, 512)
        self.TCN = TimeSeriesCNN(n_layers, input_size = 512, hidden_size = hidden_size, output_size = lang.vocab_size)


    def forward(self, images, indices, test_time = False):
        #indices size [batch_size, 512, num_words]
        #images size [batch_size, 3, 224, 224]
        features = self.feature_extractor(images)
        if test_time:
            return self.predict(features)
        else:
            R, num_captions, num_words = indices.size()
            features = features.unsqueeze(1).expand(-1, num_captions, -1, -1).contiguous()
            features = features.view(-1, 512, 1)
            indices = indices.contiguous().view(R*num_captions, num_words)
            embeds = self.embeddings(indices).permute(0,2,1)
            embeds = torch.cat([features, embeds], dim=2)
            out = self.TCN(embeds)
            return out

    
    def predict(self, features):
        #features size [batch_size, 512, 1]
        features = features[0:1,:,:]

        pred_words = features
        caption = []
        while len(caption)<50:
            word_pred = self.TCN(pred_words)[0, :, -1]
            values, indices = word_pred.topk(2)
            taken = indices[0]
            if (indices[0] == self.lang.UKN).data.cpu().numpy():
                taken = indices[1]
            elif (indices[0] == self.lang.EOS).data.cpu().numpy():
                break
            caption += [taken]
            new_embed = self.embeddings(taken.unsqueeze(0)).permute(0,2,1)
            pred_words = torch.cat([pred_words, new_embed], dim=2)
        caption = torch.stack(caption).squeeze(1)
        caption = self.lang.variable2sentence(caption)
        return caption


class GRUAnnotator(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, lang):
        super(GRUAnnotator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.lang = lang

        self.feature_extractor = FeatureExtractor()
        self.embeddings = nn.Embedding(lang.vocab_size, embedding_size)

        self.cells = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.cells.append(nn.GRUCell(embedding_size, hidden_size))
        self.dropouts.append(nn.Dropout(0.2))
        for _ in range(n_layers-1):
            self.cells.append(nn.GRUCell(hidden_size, hidden_size))
            self.dropouts.append(nn.Dropout(0.2))

        self.out_linear = nn.Linear(hidden_size, lang.vocab_size)

        self.orig_hidden = Parameter(torch.FloatTensor(hidden_size), requires_grad=False)


    def init_hidden(self, features, batch_size, num_captions):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        first = features.resize(1, batch_size, self.embedding_size)
        hidden = [first.expand(num_captions, -1, -1).contiguous().view(num_captions*batch_size, self.embedding_size)]
        for _ in range(self.n_layers-1):
            hidden_tensor = self.orig_hidden.data.zero_().unsqueeze(0).expand(num_captions*batch_size, -1)
            hidden += [Variable(hidden_tensor, requires_grad=False)]
        self.hidden = hidden


    def forward(self, images, indices, test_time = False):
        features = self.feature_extractor(images)
        if test_time:
            return self.predict(features)
        else:
            batch_size, num_captions, num_words = indices.size()
            self.init_hidden(features, batch_size, num_captions)
            features = features.unsqueeze(1).expand(-1, num_captions, -1, -1).contiguous()
            features = features.view(-1, 512, 1)
            indices = indices.contiguous().view(batch_size*num_captions, num_words)
            embeds = self.embeddings(indices).permute(1,0,2)
            embeds = torch.cat((features.permute(2,0,1), embeds), dim=0)
            words = []
            for word in embeds:
                out = self.GRU_forward(word)
                words += [out]
            caption = torch.stack(words)
            return caption.permute(1,2,0)

    def GRU_forward(self, word_embedded):
        self.hidden[0] = self.cells[0](word_embedded, self.hidden[0])
        self.hidden[0] = self.dropouts[0](self.hidden[0])
        for i in range(1, self.n_layers):
            self.hidden[i] = self.cells[i](self.hidden[i-1], self.hidden[i])
            self.hidden[i] = self.dropouts[i](self.hidden[i])
        out = self.out_linear(self.hidden[-1])
        return out

    
    def predict(self, features):
        #features size [batch_size, 512, 1]
        features = features[0:1,:,:]
        self.init_hidden(features, 1, 1)
        features = features.permute(2,0,1) #output size (1, 1, 512)
        latest_word_embedded = features[0]
        caption = []
        while len(caption)<50:
            word_pred = self.GRU_forward(latest_word_embedded)[-1]
            values, indices = word_pred.topk(2)
            taken = indices[0]
            if (indices[0] == self.lang.UKN).data.cpu().numpy():
                taken = indices[1]
            elif (indices[0] == self.lang.EOS).data.cpu().numpy():
                break
            caption += [taken]
            latest_word_embedded = self.embeddings(taken)
        caption = torch.stack(caption).squeeze(1)
        caption = self.lang.variable2sentence(caption)
        return caption

from torchvision import datasets
import re
import string
import os
import pickle
import numpy as np
  
class Lang():
    def __init__(self):
        self.EOS = 0
        self.UKN = 1
        self.EOS_string = "<EOS>"
        self.UKN_string = "<UKN>"

        if not os.path.exists("models/savedir/vocab/word2index.pkl"):
            if not os.path.exists("models/savedir/vocab/index2word.pkl"):
                self.create_dicts_from_scratch()

        with open('models/savedir/vocab/word2index.pkl', 'rb') as f:
            self.word2index = pickle.load(f)
        with open('models/savedir/vocab/index2word.pkl', 'rb') as f:
            self.index2word = pickle.load(f)

        self.vocab_size = len(self.word2index)


    def word2numpy(self, word):
        word = word.lower()
        word = re.sub('['+string.punctuation+']', '', word)
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.UKN

    
    def numpy2word(self, index):
        return self.index2word[index]

    
    def sentence2numpy(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub('['+string.punctuation+']', '', sentence).split()
        numpy_indices = [self.word2numpy(word) for word in sentence]
        return np.array(numpy_indices).astype(np.int64)

    
    def numpy2sentece(self, numpy_indices):
        caption = [self.numpy2word(index) for index in numpy_indices]
        caption[0] = caption[0].title()
        caption = " ".join(caption)
        return caption


    def sentence2variable(self, sentence, use_cuda = True, volatile = False):
        tensor = torch.from_numpy(self.sentence2numpy(sentence)).long()
        if use_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, volatile=volatile)

    
    def variable2sentence(self, variable_sentence_indices):
        indices = variable_sentence_indices.data.cpu().numpy()
        return self.numpy2sentece(indices)


    def create_dicts_from_scratch(self):
        if not os.path.exists("models/savedir/vocab/count_dict.pkl"):
            print("word/count dict not found, generating new dict from scratch, takes approx 10 min")
            self.generate_vocabulary()

        count_dict = {}
        with open("models/savedir/vocab/count_dict.pkl", 'rb') as f:
            count_dict = pickle.load(f)

        word2index = {self.EOS_string : self.EOS, self.UKN_string : self.UKN}
        index2word = {self.EOS : self.EOS_string, self.UKN : self.UKN_string}
        count = len(word2index)
        for word, word_count in count_dict.items():
            if word_count >= 5:
                word2index[word] = count
                index2word[count] = word
                count += 1
        
        with open('models/savedir/vocab/word2index.pkl', 'wb') as f:
            pickle.dump(word2index, f)
        with open('models/savedir/vocab/index2word.pkl', 'wb') as f:
            pickle.dump(index2word, f)


    def generate_vocabulary(self):
        coco_path = "/hdd/Data/MSCOCO2017/images"
        annFile = "/hdd/Data/MSCOCO2017/annotations"
        data_train = datasets.CocoCaptions(root = coco_path+"/train2017/",
                                annFile = annFile+"/captions_train2017.json")
        # #only on train data, not val data
        #data_val = datasets.CocoCaptions(root = coco_path+"/val2017/",
        #                        annFile = annFile+"/captions_val2017.json")
        all_words = []
        count_dict = {}
        for data in data_train:
            im, captions = data
            for caption in captions:
                caption = caption.lower()
                caption = re.sub('['+string.punctuation+']', '', caption).split()
                for word in caption:
                    if word in all_words:
                        count_dict[word] += 1
                    else:
                        all_words.append(word)
                        count_dict[word] = 1

        with open('models/savedir/vocab/count_dict.pkl', 'wb') as f:
            pickle.dump(count_dict, f)