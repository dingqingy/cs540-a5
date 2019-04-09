# Load X and y variable
using JLD
data = load("../data/PINS.jld")
X = data["X"]

(n,d) = size(X)

# We bave 1o digits
k = 10

# E is our list of edges:
#   - a (number of edges) by 2 matrix
#   - in column 1 you put the first node on the edge
#   - in column 2 you put the second node on the edge
#   For example, if you just want the edges 1-2 and 1-3, use:
#   - edges = [1 2;1 3]
# E = zeros(0,2) # Empty graph

# problem 1.2.3
# E = [1 2;2 3;3 4] # Chain structured graph

# problem 1.2.4
E = [1 2;1 3;1 4;2 3;2 4;3 4] # complete graph
nEdges = size(E,1)

# Compute sufficient statistics
include("UGM.jl")
(nodeStats,edgeStats) = suffStat(X,E,k)

# Initialize node and edge parameters
w = zeros(d,k)
v = zeros(k,k,nEdges)

# Fit UGM (the NLL function does brute-force inference)
include("findMin.jl")
funObj(wv) = UGM_NLL(wv,E,nodeStats,edgeStats)
wv = findMin(funObj,[w[:];v[:]])
w = reshape(wv[1:d*k],d,k)
v= reshape(wv[d*k+1:end],k,k,nEdges)

# Decoding
xDecode = UGM_Decode(w,v,E)
@show xDecode

# Inference
(Z,nodeMarg,edgeMarg) = UGM_Infer(w,v,E)
@show nodeMarg[1,:]

# Conditional inference on first states being 1,2,3.
xc = [1 2 3 0]
(Z_cond,nodeMarg_cond,edgeMarg_cond) = UGM_Infer_Cond(w,v,E,xc)
@show nodeMarg_cond[4,:]

# Sampling
n_samples = 1000
samples = UGM_Sample(w,v,E,n_samples)

count = 0
for i = 1:n_samples
	flag = true
	for j = 1:d
		if(samples[i, j]!=j)
			flag = false
		end
	end
	if(flag)
		global count += 1
	end
end

@show(count / n_samples)
# @show(samples)