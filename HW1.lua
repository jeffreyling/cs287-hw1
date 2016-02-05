-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha', 19, 'alpha for naive Bayes')
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 5, 'batch size for SGD')
cmd:option('-max_epochs', 1000, 'max # of steps for SGD')

function train_nb(nclasses, nfeatures, X, Y, alpha)
  -- Trains naive Bayes model
  alpha = alpha or 0
  local N = X:size(1)

  -- intercept
  local b = torch.histc(Y:double(), nclasses)
  b:div(b:sum())
  b:log()

  local W = torch.Tensor(nclasses, nfeatures):fill(alpha)
  local indices = torch.linspace(1, X:size(1), X:size(1)):long()
  for c = 1, nclasses do
    local counts = torch.histc(X:index(1, indices[Y:eq(c)]):double(), nfeatures)
    W[c]:add(counts)
  end
  -- zero out padding counts
  W:select(2, 1):zero()
  W:cdiv(W:sum(2):expand(W:size(1), W:size(2)))
  W:log()
  -- padding weight to zero
  W:select(2, 1):zero()

  return W, b
end

function NLL(pred, Y)
  -- Returns negative log-likelihood error.
  local probs = pred:index(2, Y:long())
  return -probs:log():sum()
end

function sgd_step(X_batch, Y_batch, W, b, eta)
    local N = X_batch:size(1)
    local z = torch.Tensor(N, nclasses):fill(0)
    local Y_hat = torch.Tensor(N, nclasses):fill(0)
    local z_grad = torch.Tensor(N, nclasses):fill(0)

    for i = 1, N do
      -- get predictions
      z[i] = W:index(2, X_batch[i]:long()):sum(2)
      -- get softmax
      local max = z[i]:max()
      Y_hat[i] = z[i]:csub(max):exp()
      Y_hat[i]:div(Y_hat[i]:sum())
      -- get gradient w.r.t. z
      z_grad[i] = Y_hat[i]:mul(-1)
      z_grad[i][Y_batch[i]] = z_grad[i][Y_batch[i]] + 1
    end

    -- collapse and increment weights
    z_grad = z_grad:mean(1):transpose(1,2):mul(eta)
    b:add(z_grad)
    W:add(z_grad:expand(W:size(1),W:size(2)))
    return W, b
end

function train_logreg(nclasses, nfeatures, X, Y, eta, batch_size, max_epochs)
  eta = eta or 0
  batch_size = batch_size or 0
  max_epochs = max_epochs or 0
  local N = X:size(1)

  -- initialize weights and intercept
  local W = torch.Tensor(nclasses, nfeatures):fill(0)
  local b = torch.Tensor(nclasses, 1):fill(0)
  local epoch = 0

  local loss = 100
  while loss > 10 and epoch < max_epochs do
    -- get batch
    local batch_indices = torch.multinomial(torch.ones(1, N), batchsize):long()
    X_batch = X:index(1, batch_indices[1])
    Y_batch = Y:index(1, batch_indices[1])

    -- update weights
    W, b = sgd_step(X_batch, Y_batch, W, b, eta)
    
    -- calculate loss
    local pred = torch.zeros(N, nclasses)
    local cross_entropy = torch.zeros(N, 1)
    for i = 1, N do
      pred[i] = W:index(2, X[i]:long()):sum(2)
      local max = pred[i]:max()
      cross_entropy[i] = pred[i]:csub(max + math.log(pred[i]:csub(max):exp():sum()))[Y[i]] * -1
    end
    loss = cross_entropy:sum()
    print(loss)
    epoch = epoch + 1
  end
  return W, b
end

function eval(X, Y, W, b, nclasses, model)
  -- Returns error from Y
  model = model or 'nb'

  local N = X:size(1)
  local pred = torch.zeros(N, nclasses)
  if model == 'nb' then
    for i = 1, N do
      pred[i] = W:index(2, X[i]:long()):sum(2)
      pred[i]:add(b)
    end
  end

  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()
  local err = argmax:eq(Y:long()):sum() / Y:size(1)
  return argmax, err
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   print('Loading data...')
   local X = f:read('train_input'):all()
   local Y = f:read('train_output'):all()
   local valid_X = f:read('valid_input'):all()
   local valid_Y = f:read('valid_output'):all()
   local test_X = f:read('test_input'):all()
   print('Data loaded.')

   -- sample for faster training
   local batch_indices = torch.multinomial(torch.ones(1, X:size(1)), 100):long()[1]
   X = X:index(1, batch_indices)
   Y = Y:index(1, batch_indices)

   --local W = torch.zeros(nclasses, nfeatures)
   --local b = torch.zeros(nclasses)

   -- Train.
   -- local W, b = train_nb(nclasses, nfeatures, X, Y, opt.alpha)
   local W, b = train_logreg(nclasses, nfeatures, X, Y, opt.eta, opt.batch_size, opt.max_epochs)

   -- Test.
   local pred, err = eval(valid_X, valid_Y, W, b, nclasses)
   print('Percent correct:', err)
end

main()
