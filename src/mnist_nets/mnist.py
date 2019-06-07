__author__ = 'joon'

import sys

sys.path.insert(0, 'src')

from lib_pytorch.baseblocks import *
from imports.basic_modules import *
from imports.ResearchTools import *
from imports.pytorch_imports import *


###

def config(control, conf, called=False):
    co = experiment_control(
        control,
        conf,
        dict(
            net=dict(
                name='mnet',
            ),
            opt=dict(
                optimiser='SGD',
                momentum=0.5,
                lr=0.01,
                epochs=100,
                batch_size=256,
            ),
            data=dict(
                dataset='mnist',
                subset='all',
            ),
            seed=1,
        )
    )

    modeldir = osp.join('cache', co.conf['exp_phase'], co.token)
    mkdir_if_missing(modeldir)

    co.conf['loc'] = dict(
        intermediatemodel=osp.join(modeldir, '%d_%d.pth.tar'),
        finalmodel=osp.join(modeldir, 'final.pth.tar'),
        testoutput=osp.join(modeldir, 'output.pkl'),
        statoutput=osp.join(modeldir, 'stat.pkl'),
    )

    if not called:
        torch.manual_seed(co.control['seed'])
        if co.conf['gpu'] is not None:
            torch.cuda.set_device(co.conf['gpu'])
        torch.cuda.manual_seed(co.control['seed'])

        if osp.isfile(co.conf['loc']['finalmodel']):
            if not co.conf['overridecache_model']:
                raise CacheFileExists("Final model %s already exists!" % (co.conf['loc']['finalmodel']))

        if osp.isfile(co.conf['loc']['testoutput']):
            if not co.conf['overridecache_output']:
                raise CacheFileExists("Output %s already exists!" % (co.conf['loc']['testoutput']))

    return co


def mnist_data_transform(data, direction='forward'):
    if direction == 'forward':
        return (data - 0.1307) / 0.3081
    elif direction == 'backward':
        return data * 0.3081 + 0.1307
    else:
        raise ValueError('Data transformation direction is either forward or backward.')


class return_act(nn.Module):
    def __init__(self, act):
        super(return_act, self).__init__()
        if act == 'relu':
            self.actfunc = nn.ReLU()
        elif act == 'elu':
            self.actfunc = nn.ELU()
        elif act == 'prelu':
            self.actfunc = nn.PReLU()
        elif act == 'tanh':
            self.actfunc = nn.Tanh()
        else:
            raise ValueError('Activation type should be one of {relu, elu, prelu, tanh}.')

    def forward(self, x):
        return self.actfunc(x)


def return_drop(drop):
    if drop == 'normal':
        dropfunc = F.dropout
    else:
        raise ValueError('Dropout type must be "normal".')
    return dropfunc


class mnet(nn.Module):
    def __init__(self, control):
        super(mnet, self).__init__()
        self.control = control
        self.act = return_act(control['net']['act'])

        self.ks = self.control['net']['ks']
        self.conv1 = nn.Conv2d(1, 10, kernel_size=self.ks)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=self.ks)
        if 'max' in self.control['net']['pool']:
            stride = int(self.control['net']['pool'].split('_')[1])
            self.pool = lambda a: F.max_pool2d(a, stride)
            poolfactor = 2
        else:
            self.pool = lambda a: a
            poolfactor = 1
        conv_iter = []
        for n in range(self.control['net']['n_conv'] - 2):
            conv_iter.append(convblock(20, 20, self.ks, padding=int((self.ks - 1) / 2)))
        self.conv_iter = nn.Sequential(*conv_iter)

        self.fcfeatdim = int((
                             (
                                 (
                                     (
                                         (28 - self.ks + 1) // poolfactor
                                     ) - self.ks + 1) // poolfactor
                             ) ** 2
                         ) * 20)

        self.fc1 = nn.Linear(self.fcfeatdim, 50)
        fc_iter = []
        for n in range(self.control['net']['n_fc'] - 2):
            fc_iter.append(linearblock(50, 50, self.control['net']['drop']))
        self.fc_iter = nn.Sequential(*fc_iter)
        self.fc_final = nn.Linear(50, 10)

    def forward(self, x):
        x = self.act.forward(self.pool(self.conv1(x)))
        x = self.act.forward(self.pool(self.conv2(x)))
        x = self.conv_iter(x)
        x = x.view(-1, self.fcfeatdim)
        x = self.act.forward(self.fc1(x))
        if self.control['net']['drop'] == 'normal':
            x = F.dropout(x, training=self.training)
        x = self.fc_iter(x)
        x = self.fc_final(x)
        return F.log_softmax(x)


class nin(nn.Module):
    def __init__(self, control):
        super(nin, self).__init__()
        self.act = return_act(control['net']['act'])
        self.drop = return_drop(control['net']['drop'])
        self.conv1 = nn.Conv2d(1, 96, 5, stride=1, padding=2)
        self.cccp1 = nn.Conv2d(96, 64, 1, stride=1, padding=0)
        self.cccp2 = nn.Conv2d(64, 48, 1, stride=1, padding=0)

        self.conv2 = nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.cccp3 = nn.Conv2d(128, 96, 1, stride=1, padding=0)
        self.cccp4 = nn.Conv2d(96, 48, 1, stride=1, padding=0)

        self.conv3 = nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.cccp5 = nn.Conv2d(128, 96, 1, stride=1, padding=0)
        self.cccp6 = nn.Conv2d(96, 10, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.act.forward(self.conv1(x))
        x = self.act.forward(self.cccp1(x))
        x = self.act.forward(self.cccp2(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = self.drop(x, training=self.training)

        x = self.act.forward(self.conv2(x))
        x = self.act.forward(self.cccp3(x))
        x = self.act.forward(self.cccp4(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = self.drop(x, training=self.training)

        x = self.act.forward(self.conv3(x))
        x = self.act.forward(self.cccp5(x))
        x = self.act.forward(self.cccp6(x))
        x = F.avg_pool2d(x, 6)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x)


def load_model(control, gpu=True):
    if control['net']['name'] == 'mnet':
        model = mnet(control)
    elif control['net']['name'] == 'nin':
        model = nin(control)
    else:
        raise ValueError('Network name should be mnet.')

    if gpu is not None:
        model.cuda()

    return model


def collect_sampler(subset, original_seed):
    N = 60000
    np.random.seed(20)
    if subset == 'all':
        sampler = np.arange(N)
        np.random.shuffle(sampler)
    elif 'half' in subset:
        _, _t = subset.split('_')
        _t = int(_t)
        sampler = np.arange(N)
        np.random.shuffle(sampler)
        sampler = sampler[_t * 30000:(_t + 1) * 30000]
    elif 'quarter' in subset:
        _, _t = subset.split('_')
        _t = int(_t)
        sampler = np.arange(N)
        np.random.shuffle(sampler)
        sampler = sampler[_t * 15000:(_t + 1) * 15000]
    else:
        raise ValueError('Subset should be one of {all, half_#, quarter_#}.')

    np.random.seed(original_seed)
    return sampler.tolist()


def train(co):
    assert (co.conf['exp_phase'] == 'modelzoo-mnist')
    kwargs = {'num_workers': 1, 'pin_memory': True}
    sampler_list = collect_sampler(co.control['data']['subset'], original_seed=co.control['seed'])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        sampler=torch.utils.data.sampler.SubsetRandomSampler(sampler_list),
        batch_size=co.control['opt']['batch_size'], shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=co.control['opt']['batch_size'], shuffle=True, **kwargs)

    model = load_model(co.control, gpu=co.conf['gpu'])
    if co.control['opt']['optimiser'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=co.control['opt']['lr'], momentum=co.control['opt']['momentum'])
    elif co.control['opt']['optimiser'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=co.control['opt']['lr'])
    elif co.control['opt']['optimiser'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=co.control['opt']['lr'],
                                  momentum=co.control['opt']['momentum'])
    else:
        raise ValueError('Optimiser should be one of {SGD, ADAM, RMSprop}.')

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if co.conf['gpu'] is not None:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % co.conf['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))

    def test():
        model.eval()
        test_loss = 0.0
        correct = 0.0
        for data, target in test_loader:
            if co.conf['gpu'] is not None:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += float(pred.eq(target.data.view_as(pred)).sum().item())

        acc = 100. * correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            acc))

        return acc

    for epoch in range(1, co.control['opt']['epochs'] + 1):
        train(epoch)
        acc = test()

    if co.conf['save']:
        torch.save({
            'control': co.control,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, co.conf['loc']['finalmodel'])
        save_to_cache(dict(acc=acc), co.conf['loc']['statoutput'])

    return


def compute_features(co):
    assert (co.conf['exp_phase'] == 'modelzoo-mnist')
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=False, **kwargs)

    checkpoint = torch.load(co.conf['loc']['finalmodel'], map_location=lambda storage, loc: storage)

    model = load_model(co.control)
    optimizer = optim.SGD(model.parameters(), lr=co.control['opt']['lr'], momentum=co.control['opt']['momentum'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    def test():
        ip, op = [], []
        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            ip.append(data.data.cpu().numpy())
            op.append(output.data.cpu().numpy())
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * correct / len(test_loader.dataset)
        return ip, op, acc

    ip, op, acc = test()

    if co.conf['save']:
        save_to_cache(dict(ip=ip, op=op, acc=acc), co.conf['loc']['testoutput'])
        save_to_cache(dict(acc=acc), co.conf['loc']['statoutput'])

    return dict(acc=acc)


def training(control, conf):
    co = config(control, conf)
    with co:
        train(co)


def testing(control, conf):
    co = config(control, conf)
    with co:
        res = compute_features(co)
    return res
