import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras import optimizers
import keras.backend as K
import random
from tensorflow.keras import metrics, losses
import numpy as np
import argparse
from sklearn.metrics import pairwise
from scipy.spatial import distance
from sklearn.preprocessing import normalize

# NEAR-COPY OF ABSent MODEL (Fu et al., 2020)
parser = argparse.ArgumentParser()
parser.add_argument('--lang1', required=True, type=str, help='L1 embeddings')
parser.add_argument('--lang2', required=True, type=str, help='L2 embeddings')
parser.add_argument('--dictionary', required=True, type=str, help='L1-L2 dictionary')
parser.add_argument('--evaldict', required=True, type=str, help='L1-L2 evaluation dictionary')
parser.add_argument('--output', required=False, type=str, help='filename to save mapping model to (optional)')
parser.add_argument('--predict', required=False, type=str, help='output file for predictions (optional)')
parser.add_argument('--layers', required=False, type=int, default=9, help='how many layers in mapping model')
parser.add_argument('--bs', required=False, type=int, default=128, help='batch size')
parser.add_argument('--lr', required=False, type=float, default=0.002, help='learning rate')
parser.add_argument('--decay', required=False, type=float, default=0, help='learning rate decay')
parser.add_argument('--epoch', type=int, default=3, help='num of epochs')
parser.add_argument('--iters', required=False, type=int, help='number of iterations to train, overrides --epoch')
parser.add_argument('--layer0', action='store_true', help='enable if mapping non-contextual layer')
args = parser.parse_args()

def htanh(a):
    return K.maximum(-1.0, K.minimum(1.0, a))
    
def load_embeddings(embfile):
    with open(embfile, 'r') as reader:
        emb = {}
        header = reader.readline()
        header = header.strip().split()
        dim = int(header[1])
        for line in reader:
            line = line.strip().split()
            if args.layer0:
                word = ''.join([c for c in line[0] if not c.isdigit()])
            else:
                word = line[0]
            if len(line) == dim+1:
                emb[word] = [float(i) for i in line[1:]]
    return emb, dim

x_tr = []
x_ev = []
y_tr = []
y_ev = []
y_short = []

# Load lang1 embs, lang2 embs and dictionary
print("Loading embeddings")
emb1, emb1_dim = load_embeddings(args.lang1)
emb2, emb2_dim = load_embeddings(args.lang2)
assert emb1_dim == emb2_dim

print("Loading dictionaries")
with open(args.dictionary, 'r') as reader:
    dictionary = []
    l1set = set()
    for line in reader:
        lin = line.strip().split()
        word = lin[0]
        if not word in l1set:
            if args.layer0:
                lin = [''.join([c for c in word if not c.isdigit()]) for word in lin]
            dictionary.append(lin)
            l1set.add(word)
    l1set = None
    
with open(args.evaldict, 'r') as reader:
    dictionary_ev = []
    for line in reader:
        if args.layer0:
            lin = ''.join([c for c in line if not c.isdigit()])
        else:
            lin = line
        dictionary_ev.append(lin.strip().split())     
        


print("Preparing train and eval data")
# Match x_tr (lang1) to y_tr (lang2) via dictionary, set some off for eval
dictionary_tr = []
for entry in dictionary:
    if entry[0] in emb1 and entry[1] in emb2:
        x_tr.append(emb1[entry[0]])
        y_tr.append(emb2[entry[1]])
        dictionary_tr.append((entry[0],entry[1]))
    #if len(entry[1]) < 4 and entry[1] in emb2:
    #    y_short.append(emb2[entry[1]])
        
dictionary_pr = []
for entry in dictionary_ev:
    if entry[0] in emb1 and entry[1] in emb2:
        x_ev.append(emb1[entry[0]])
        y_ev.append(emb2[entry[1]])
        dictionary_pr.append((entry[0],entry[1]))


for entry in emb2:
    if len(entry) < 4:
        y_short.append(emb2[entry])
        
dictionary_x = []
dictionary_y = []
x_all = []
y_all = []
for entry in emb2:
    if len(entry) > 1:
        dictionary_y.append(entry)
        y_all.append(emb2[entry])
for entry in emb1:
    x_all.append(emb1[entry])
    dictionary_x.append(entry)

def prepare(data):
    data = np.array(data).astype(np.float32)
    print(np.shape(data))
    return data
        
def calcscore(y_pr, y_ev):
    srazlika = 0
    scosine = 0
    for i in range(len(y_pr)):
        razlika = [abs(float(y_pr[i][j]) - float(y_ev[i][j])) for j in range(len(y_pr[i]))]
        prediction = [float(y_pr[i][j]) for j in range(len(y_pr[i]))]
        evalset = [float(y_ev[i][j]) for j in range(len(y_ev[i]))]
        srazlika += sum(razlika)/len(razlika)
        scosine += distance.cosine(prediction,evalset)
    return srazlika/len(y_pr), scosine/len(y_pr)

def find_closest(x, ys):
    x = np.asarray(x)
    ys = np.asarray(ys)
    return np.argmax(ys.dot(x))
def find_closest_mult(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs = normalize(xs)
    ys = normalize(ys)
    sims = xs@ys.transpose()
    top10s = [s.argsort()[-10:][::-1] for s in sims]
    return np.argmax(xs@ys.transpose(), axis=1), top10s # for each x in xs, find closest among ys

    
emb2 = None
dictionary = None
amount=6
trsize=5000000

x_ev = prepare(x_ev)
y_ev = prepare(y_ev)
x_tr = prepare(x_tr[:trsize])
y_tr = prepare(y_tr[:trsize])


def make_oneway_generator(generator_name):
    input_layer = Input(shape=(emb1_dim,), dtype="float32", name="gen-input")
    layer1 = Dense(2048, activation="relu", name="gen-1") (input_layer)
    batchnorm1 = BatchNormalization() (layer1)
    layer2 = Dense(4096, activation="relu") (batchnorm1)
    batchnorm2 = BatchNormalization() (layer2)
    layer3 = Dense(2048, activation="relu") (batchnorm2)
    out = Dense(emb1_dim, activation="tanh") (layer3)
    model = Model(inputs=input_layer, outputs=out, name=generator_name)
    return model

def make_discriminator(disc_name):
    input_y = Input(shape=(emb1_dim,), dtype="float32", name="disc-y")
    input_x = Input(shape=(emb1_dim,), dtype="float32", name="disc-x")
    layer0 = Concatenate(name="disc-concat")([input_y, input_x])
    layer1 = Dense(2048) (layer0)
    lrelu1 = LeakyReLU(alpha=0.2) (layer1)
    layer2 = Dense(4096) (lrelu1)
    lrelu2 = LeakyReLU(alpha=0.2) (layer2)
    layer3 = Dense(2048) (lrelu2)
    out = Dense(1, activation="sigmoid") (layer3)
    model = Model(inputs=[input_x, input_y], outputs=out, name=disc_name)
    return model

def make_bigenerator():
    input_x = Input(shape=(emb1_dim,), dtype="float32")
    input_y = Input(shape=(emb2_dim,), dtype="float32")
    generator1 = make_oneway_generator("gen1")
    generator2 = make_oneway_generator("gen2")
    x_mapped = generator1(input_x)
    y_mapped = generator2(input_y)
    return Model(inputs=[input_x, input_y], outputs=[x_mapped, y_mapped], name="bigenerator")
    

adam = optimizers.Adam(lr=args.lr, decay=args.decay)
x_real = Input(shape=(emb1_dim,), dtype="float32", name="X_real")
y_real = Input(shape=(emb2_dim,), dtype="float32", name="Y_real")
generator = make_bigenerator() 
print("Generator made")
x_mapped, y_mapped = generator([x_real,y_real])
print("Mapped layer generated")
discriminator_valid = make_discriminator("discvalid")
discriminator_domain = make_discriminator("discdomain")
discriminator_valid.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
discriminator_domain.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print("Discriminator compiled")
discriminator_valid.trainable = False
discriminator_domain.trainable = False

validated = discriminator_valid([x_mapped, y_mapped])
domainx = discriminator_domain([x_real, x_mapped])
domainy = discriminator_domain([y_real, y_mapped])
print("Discriminator output generated")
gan = Model(inputs=[x_real,y_real], outputs=[validated, domainx, domainy, x_mapped, y_mapped])
gan.compile(optimizer=adam, loss={'discvalid': 'binary_crossentropy', 'discdomain': 'binary_crossentropy', 'bigenerator': losses.cosine_similarity}, loss_weights={'discvalid': 1.0, 'discdomain': 1.0, 'bigenerator': 0.4})
print("GAN compiled")
if args.predict:
    losslog = args.predict+'.loss.log'
else:
    losslog = 'loss.log'
#model.compile(optimizer=adam, loss="cosine_proximity", metrics=[metrics.mse, metrics.cosine_proximity])
real = np.ones(args.bs)*1.00
fake = np.ones(args.bs)*0.00
num_examples = len(y_tr)
if args.iters:
    iters = args.iters
else:
    iters = int(args.epoch * num_examples // args.bs)
print("Start training for",iters,"iterations.")
with open(losslog, 'w') as losswriter:
    losswriter.write('#iter\ttrain_Dloss\ttrain_GAN-Dloss\ttrain_GAN-Gloss\teval_GAN-Dloss\teval_GAN-Gloss\n')
    curr = 0
    for it in range(iters):
        # train discriminator
        #curr = (it%args.epoch)*args.bs
        curr += args.bs
        if curr >= num_examples:
            curr = 0
        batchsize = min(args.bs, num_examples-curr)
        d_valid_loss1 = discriminator_valid.train_on_batch([x_tr[curr:curr+args.bs],y_tr[curr:curr+args.bs]], real[:batchsize])
        #x_fake = np.random.uniform(-1, 1, (batchsize//2,emb1_dim))
        #y_fake = generator.predict(x_fake)
        Gx, Gy = generator.predict([x_tr[curr:curr+args.bs], y_tr[curr:curr+args.bs]])
        x_rand = x_tr[np.random.randint(0, x_tr.shape[0], batchsize)]
        y_rand = y_tr[np.random.randint(0, y_tr.shape[0], batchsize)]
        d_valid_loss2 = discriminator_valid.train_on_batch([x_rand, y_rand], fake[:batchsize])
        d_valid_loss3 = discriminator_valid.train_on_batch([x_tr[curr:curr+args.bs], Gx], fake[:batchsize])
        d_valid_loss4 = discriminator_valid.train_on_batch([y_tr[curr:curr+args.bs], Gy], fake[:batchsize])
        d_valid_loss = np.add(0.25*np.add(d_valid_loss3,d_valid_loss4), 0.25*np.add(d_valid_loss1,d_valid_loss2))

        d_domain_loss1 = discriminator_domain.train_on_batch([x_tr[curr:curr+args.bs], Gx], real[:batchsize])
        d_domain_loss2 = discriminator_domain.train_on_batch([y_tr[curr:curr+args.bs], Gy], fake[:batchsize])
        d_domain_loss = 0.5*np.add(d_domain_loss1, d_domain_loss2)
        
        # train generator
        g_loss = gan.train_on_batch([x_tr[curr:curr+args.bs], y_tr[curr:curr+args.bs]], [real[:batchsize], real[:batchsize], fake[:batchsize], y_tr[curr:curr+args.bs], x_tr[curr:curr+args.bs]])
        if it%300 == 0:
            try:
                print("[Iteration %d/%d] [D_valid loss: %f, acc: %f] [D_domain loss: %f, acc: %f] [G loss_bce: %f, loss_cos: %f]" % (it+1, iters, d_valid_loss[0], d_valid_loss[1], d_domain_loss[0], d_domain_loss[1], g_loss[0], g_loss[1]))
            except:
                print("[Iteration %d/%d]" % (it+1, iters))
            eval_loss = gan.test_on_batch([x_ev,y_ev], [np.ones(len(y_ev)), np.ones(len(y_ev)), np.zeros(len(y_ev)), y_ev, x_ev])
            print("[EVAL] [loss_bce: %f, loss_cos: %f]" % (eval_loss[0], eval_loss[1]))
            losswriter.write("%d\t%f\t%f\t%f\t%f\t%f\n" % (it+1, d_valid_loss[0]+d_domain_loss[0], g_loss[0], g_loss[1], eval_loss[0], eval_loss[1]))
        # EARLY STOPPING
        if it/iters > 1.0:
            if eval_loss[1] >= 10*g_loss[1] or eval_loss[0] >= 0:
                print("TRAINING STOPPED EARLY")
                losswriter.write("%d\t%f\t%f\t%f\t%f\t%f\n" % (it+1, d_valid_loss[0]+d_domain_loss[0], g_loss[0], g_loss[1], eval_loss[0], eval_loss[1]))
                break


# CALC PREDICTIONS
y_pr, x_pr = generator.predict([x_ev,y_ev], batch_size=128)
#y_tr_pr = generator.predict(np.asarray(x_tr), batch_size=128)
#print(np.shape(y_pr))
#print(np.shape(y_tr_pr))

# DICTIONARY INDUCTION PREDICTIONS
def write_predictions(filename, x_predicted, y_predicted, dict_predict):
    with open(filename, 'w') as writer:
        writer.write('#x_real y_real x_predicted y_predicted\n')
        print('finding closest words to predictions')
        xpr_indices, top10xpr = find_closest_mult(x_predicted, x_all)
        ypr_indices, top10ypr = find_closest_mult(y_predicted, y_all)
        for i,x in enumerate(x_predicted):
            writer.write(" ".join(dict_predict[i])+" "+dictionary_x[xpr_indices[i]]+" "+dictionary_y[ypr_indices[i]]+"\n")
    print('writing top 10')
    with open(filename+'.top10', 'w') as writer:
        x_acc = {1: 0, 3: 0, 5: 0, 10: 0}
        y_acc = {1: 0, 3: 0, 5: 0, 10: 0}
        for i in range(len(x_predicted)):
            x_words = [dictionary_x[t] for t in top10xpr[i]]
            y_words = [dictionary_y[t] for t in top10ypr[i]]
            for k in x_acc:
                if dict_predict[i][0] in x_words[:k]:
                    x_acc[k] += 1
                if dict_predict[i][1] in y_words[:k]:
                    y_acc[k] += 1
        writer.write("direction\tacc@1\tacc@3\tacc@5\tacc@10\ny->x")
        for k in x_acc:
            writer.write("\t"+str(x_acc[k]/len(x_predicted)))
        writer.write("\nx->y")
        for k in y_acc:
            writer.write("\t"+str(y_acc[k]/len(y_predicted)))
        writer.write('\n')


if args.output:
    generator.save(args.output+'.h5')

if args.predict:
    write_predictions(args.predict, x_pr, y_pr, dictionary_pr)
    #write_predictions(args.predict+'.ontrain', x_tr, y_tr_pr, dictionary_tr)

x_tr = None
x_ev = None
y_tr = None
y_ev = None

    #l1vectors = np.array(list(emb1.values()))
    #l1keys = list(emb1.keys())
    #emb1 = None
    #y_predict = model.predict(np.array(l1vectors), batch_size=128)
    #with open(args.output, 'w') as writer:
    #    writer.write(header.strip()+'\n')
    #    for word in zip(l1keys, y_predict):
    #        writer.write(word[0]+' '+' '.join([str(j) for j in word[1]])+'\n')

