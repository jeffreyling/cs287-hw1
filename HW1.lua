-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha', 1, 'alpha for naive Bayes')
cmd:option('-eta', 0.001, 'learning rate for SGD')
cmd:option('-batch_size', 50, 'batch size for SGD')
cmd:option('-max_epochs', 1000, 'max # of steps for SGD')

function train_nb(nclasses, nfeatures, X, Y, alpha)
  -- Trains naive Bayes model
  alpha = alpha or 0
  local N = X:size(1)

  -- intercept
  local b = torch.histc(Y:double(), nclasses)
  local b_logsum = torch.log(b:sum())
  b:log():csub(b_logsum)

  local W = torch.Tensor(nclasses, nfeatures):fill(alpha)
  local indices = torch.linspace(1, X:size(1), X:size(1)):long()
  for c = 1, nclasses do
    local counts = torch.histc(X:index(1, indices[Y:eq(c)]):double(), nfeatures)
    W[c]:add(counts)
  end
  -- zero out padding counts
  W:select(2, 1):zero()
  local W_logsum = W:sum(2):log()
  print(W_logsum)
  W:log():csub(W_logsum:expand(W:size(1), W:size(2)))
  -- padding weight to zero
  W:select(2, 1):zero()

  return W, b
end

function NLL(pred, Y)
  -- Returns negative log-likelihood error.
  local N = Y:size(1)
  local err = 0
  for i = 1, N do
    err = err - torch.log(pred[i][Y[i]])
  end
  return err
end

function linear(X, W, b)
  -- performs y = softmax(x*W + b)
  local N = X:size(1)
  local k = X:size(2)
  local z = torch.zeros(N, W:size(1))
  for i = 1, N do
    -- get predictions
    local feats = torch.zeros(W:size(2), 1)
    feats:indexFill(1, X[i]:long(), 1)
    feats:squeeze()
    z[i] = W[{{}, feats:long()}]:sum(2)
    z[i]:add(b)
  end
  -- get softmax
  local max = z:max(2)
  local logsoftmax = z:clone()
  logsoftmax:csub(max:expand(logsoftmax:size(1), logsoftmax:size(2)))
  logsoftmax = torch.exp(logsoftmax):sum(2):log()
  logsoftmax:add(max:expand(logsoftmax:size(1), logsoftmax:size(2)))
  local Y_hat = z:clone()
  Y_hat:csub(logsoftmax:expand(Y_hat:size(1), Y_hat:size(2)))
  Y_hat:exp()
  return Y_hat, z
end

function sgd_grad(X_batch, Y_batch, W, b)
    local N = X_batch:size(1)

    local Y_hat, z = linear(X_batch, W, b)
    -- get gradient w.r.t. z
    local z_grad = Y_hat:clone()
    for i = 1, N do
      z_grad[i][Y_batch[i]] = Y_hat[i][Y_batch[i]] - 1
    end

    -- collapse and compute W, b grads
    z_grad = z_grad:mean(1):squeeze()
    local b_grad = z_grad:clone()
    local W_grad = torch.zeros(nclasses, nfeatures)
    for i = 1, N do
      W_grad:indexAdd(2, X_batch[i]:long(), z_grad:view(z_grad:nElement(), 1):expand(nclasses, X_batch[i]:size(1)))
    end
    W_grad:div(N)

    return W_grad, b_grad
end

function train_logreg(nclasses, nfeatures, X, Y, eta, batch_size, max_epochs)
  eta = eta or 0
  batch_size = batch_size or 0
  max_epochs = max_epochs or 0
  local N = X:size(1)

  -- initialize weights and intercept
  local W = torch.zeros(nclasses, nfeatures)
  local b = torch.zeros(nclasses)
  local epoch = 0

  local loss = 100
  while loss > 10 and epoch < max_epochs do
    -- get batch
    local batch_indices = torch.multinomial(torch.ones(N), batch_size, false):long()
    local X_batch = X:index(1, batch_indices)
    local Y_batch = Y:index(1, batch_indices)

    -- get gradients
    local W_grad, b_grad = sgd_grad(X_batch, Y_batch, W, b)

    -- numerical grads
    --local eps = 1e-5
    --local y1 = linear(X_batch, W, b + torch.Tensor{0,eps,0,0,0})
    --local y2 = linear(X_batch, W, b - torch.Tensor{0,eps,0,0,0})
    --print((NLL(y1, Y_batch) - NLL(y2, Y_batch)) / (2 * eps * batch_size))
    --print(b_grad)
    --io.read()

    -- update weights
    W:csub(W_grad:mul(eta))
    b:csub(b_grad:mul(eta))
    
    -- zero padding
    W:select(2, 1):zero()

    -- calculate loss
    local pred = linear(X, W, b)
    local loss = NLL(pred, Y)
    print(loss)

    --local cross_entropy = torch.zeros(N)
    --for i = 1, N do
      --pred[i] = W:index(2, X[i]:long()):sum(2)
      --local max = pred[i]:max()
      --cross_entropy[i] = pred[i]:csub(max + math.log(pred[i]:csub(max):exp():sum()))[Y[i]] * -1
    --end
    --loss = cross_entropy:sum()
    --print(loss)
    epoch = epoch + 1
  end
  print('Trained', epoch, 'epochs')
  return W, b
end

function eval(X, Y, W, b, nclasses)
  -- Returns error from Y
  local _, pred = linear(X, W, b)
  
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()
  local err = argmax:eq(Y:long()):sum() / Y:size(1)
  return argmax, err
end

function test()
  local X = torch.range(1, 12):reshape(3, 4)
  local W = torch.range(1, 24):reshape(2, 12)
  local b = torch.Tensor{1, 2}
  print(X, W, b)
  local y, z = linear(X, W, b)
  print(y, z)
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

   -- Train.
   local W, b
   if opt.classifier == 'nb' then
     W, b = train_nb(nclasses, nfeatures, X, Y, opt.alpha)
   elseif opt.classifier == 'logreg' then
     -- sample for faster training
     --local batch_indices = torch.multinomial(torch.ones(X:size(1)), 100, false):long()
     --X = X:index(1, batch_indices)
     --Y = Y:index(1, batch_indices)
     W, b = train_logreg(nclasses, nfeatures, X, Y, opt.eta, opt.batch_size, opt.max_epochs)
   end

   -- Test.
   local pred, err = eval(valid_X, valid_Y, W, b, nclasses)
   print('Percent correct:', err)
end

main()
