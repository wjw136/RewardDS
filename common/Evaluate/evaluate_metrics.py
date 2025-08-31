from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
import mauve 
from transformers import AutoModel, AutoTokenizer
from mauve.utils import featurize_tokens_from_model
import torch
import codecs
import os
import subprocess
from bert_score import score
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from rouge_chinese import Rouge


class Chinese_Meter():
    def __init__(self, tokenizer=None):
        self.rouge_scorer = Rouge()
    
    def compute_rouge(self, reference, candidate):
        # 计算 ROUGE 分数
        single_score = self.rouge_scorer.get_scores(reference, candidate)
        return single_score[0]['rouge-1']['f'], single_score[0]['rouge-2']['f'], single_score[0]['rouge-l']['f']

    def compute_bleu(self, reference, candidate):
        # 计算 BLEU 分数
        reference_tok = nltk.word_tokenize(reference)
        candidate_tok = nltk.word_tokenize(candidate)
        return 0.0

#=========================================
# bleu: n-gram precision
# rouge: n-gram recall
# meteor: + 整个语料库上的准确率和召回率
# chrF++: CHRF指标从字符级别对译文质量进行评估
# ter: 译文编辑率
# bert: 从BERT模型的输出中提取特征，然后计算相似度分数
# bleurt: 从训练过的BERT模型的输出中提取特征，然后计算相似度分数
# mauve: 用散度边界测量神经文本和人工文本间的差距(gpt)
#=========================================

class Meter():
    def __init__(self, tokenizer=None):
        # 初始化各个指标需要的模型等
        if tokenizer is None:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, tokenizer=tokenizer)
        self.smoothing = SmoothingFunction()
        self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        self.gpt_model_path = ""
        self.meteor_path = ""
        self.bleurt_checkpoint = ""
        self.bert_model_path = ""

    def compute_rouge(self, reference, candidate):
        # 计算 ROUGE 分数
        single_score = self.rouge_scorer.score(reference, candidate)
        return single_score['rouge1'].fmeasure, single_score['rouge2'].fmeasure, single_score['rougeL'].fmeasure

    def compute_bleu_corpus(self, references, candidates):
        # 计算 BLEU 分数
        multi_references_tok = [[nltk.word_tokenize(reference)] for reference in references]
        candidates_tok = [nltk.word_tokenize(candidate) for candidate in candidates]
        return corpus_bleu(multi_references_tok, candidates_tok)
    
    def compute_bleu(self, reference, candidate):
        # 计算 BLEU 分数
        reference_tok = nltk.word_tokenize(reference)
        candidate_tok = nltk.word_tokenize(candidate)
        return sentence_bleu([reference_tok], candidate_tok, smoothing_function=self.smoothing.method3)

    def compute_mauve(self, references, candidates):
        #! 最好是全部生成完计算
        # 计算 MAUVE 分数
        self.model = AutoModel.from_pretrained(self.gpt_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_path)
        referecens_tok = [self.tokenizer.encode(item, return_tensors='pt', truncation=True, max_length=1024) for item in references]
        candidates_tok = [self.tokenizer.encode(item, return_tensors='pt', truncation=True, max_length=1024) for item in candidates]
        references_features = featurize_tokens_from_model(self.model, referecens_tok, batch_size=1, name="references")
        candidates_features = featurize_tokens_from_model(self.model, candidates_tok, batch_size=1, name="candidates")
        mauve_results = mauve.compute_mauve(p_features=references_features, q_features=candidates_features, verbose=True)
        self.model.cpu()
        return mauve_results.mauve
    
    def meteor_score(self, references, candidates, num_refs, lng='en'):
        candi_tmp, refs_tmp = 'candidates_meteor', 'reference_meteor'

        with codecs.open(candi_tmp, 'w', 'utf-8') as f:
            f.write('\n'.join(candidates))

        with codecs.open(refs_tmp, 'w', 'utf-8') as f:
            f.write('\n'.join(references))
        try:
            command = 'java -Xmx2G -jar {0} '.format(self.meteor_path)
            command += '{0} {1} -l {2} -norm -r {3}'.format(candi_tmp, refs_tmp, lng, 1)
            result = subprocess.check_output(command, shell=True)
            meteor = result.split(b'\n')[-2].split()[-1]
        except:
            print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
            meteor = -1
        try:
            os.remove(candi_tmp)
            os.remove(refs_tmp)
        except:
            pass
        return float(meteor)

    def compute_bleurt(self, references, candidates):
        config = BleurtConfig.from_pretrained(self.bleurt_checkpoint)
        model = BleurtForSequenceClassification.from_pretrained(self.bleurt_checkpoint).to(self.device)
        tokenizer = BleurtTokenizer.from_pretrained(self.bleurt_checkpoint)

        model.eval()
        res_list = []
        with torch.no_grad():
            for ref, can in zip(references, candidates):
                inputs = tokenizer(ref, can, padding='longest', return_tensors='pt').to(self.device)
                res = model(**inputs).logits.flatten().tolist()
                res_list.append(res[0])
        print(res_list)
        # [0.9604414105415344, 0.8080050349235535]
        return res_list

    def compute_bert_score(self, references, hypothesis, lng='en'):
        bert_model = AutoModel.from_pretrained(self.bert_model_path).to(self.device)
        bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_path)
        for i, refs in enumerate(references):
            references[i] = [references[i]]
        # try:
        P, R, F1 = score(hypothesis, references, lang=lng, model=bert_model, tokenizer=bert_tokenizer , device=self.device)
        P, R, F1 = list(P), list(R), list(F1)
        F1 = float(sum(F1) / len(F1))
        P = float(sum(P) / len(P))
        R = float(sum(R) / len(R))
        # print("AAAAAAAAA")
        # except Exception as e:
        #     print(e)
        #     P, R, F1 = 0, 0, 0
        return P, R, F1

if __name__ == '__main__':
    # print(compute_mauve("", ""))
    # print(compute_bleu("fasdfa", "i love"))
    # print(compute_rouge("People with weakened immune systems are at higher risk for pneumonia. They include patients who are immunocompromised due to chronic underlying conditions such as diabetes, hypertension, or kidney disease, as well as those who are taking medications that affect their immune system function.<|im_end|>", 
    #                     "Sure! Pneumonia can be difficult to diagnose in people with weakened immune systems because their symptoms may be different. Some common signs of pneumonia can be coughing, chest pain, and fever. However, people with weakened immune systems may not have these symptoms. Doctors may use chest x-rays or blood tests to diagnose pneumonia in these cases.<|im_end|>ve"))

    meter = Meter()

    references = ['I love you', 'The weather is good', "how are you"]
    predictions = ['I love he', 'The weather is bad', "how are me"]
    # print(meter.compute_bert_score(references, predictions))

    # print(meter.compute_bleu("i like", "i love"))
    # meter.compute_bleurt(references, predictions)
    print(meter.compute_rouge("People with weakened immune systems are at higher risk for pneumonia. They include patients who are immunocompromised due to chronic underlying conditions such as diabetes, hypertension, or kidney disease, as well as those who are taking medications that affect their immune system function.<|im_end|>", 
                        "Sure! Pneumonia can be difficult to diagnose in people with weakened immune systems because their symptoms may be different. Some common signs of pneumonia can be coughing, chest pain, and fever. However, people with weakened immune systems may not have these symptoms. Doctors may use chest x-rays or blood tests to diagnose pneumonia in these cases.<|im_end|>ve"))


