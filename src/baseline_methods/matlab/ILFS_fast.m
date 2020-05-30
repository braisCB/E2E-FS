function [RANKED, WEIGHT] = ILFS_fast(X, Y , TT, verbose )
%  Infinite Latent Feature Selection - ICCV 2017
%  Accepted at the IEEE International Conference on Computer Vision (ICCV), 2017, Venice. Preprint copy
%  =========================================================================
%   Reference   : Infinite Latent Feature Selection
%   Author      : Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli
%   Link        : preprint: https://arxiv.org/abs/1707.07538
%   ProjectPage : http://giorgioroffo.uk
%  =========================================================================
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV17, 
% author={Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli}, 
% booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
% title={Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach}, 
% year={2017}, 
% month={Oct}}
%  ------------------------------------------------------------------------
if (nargin < 3)
    verbose = 0;
    TT = 3;
end
if (nargin < 4)
    verbose = 0;
end

A = LearningGraphWeights(X, Y, TT, verbose );

priori_len = ceil(max( A*ones(length(A),1)))/size(X,2);

factor = 0.99;
 
% assert(factor<1);
%% 4) Letting paths tend to infinite: Inf-FS Core
if (verbose)
    fprintf('4) Letting paths tend to infinite \n');
end
I = eye( size( A ,1 )); % Identity Matrix
rho = max(eig(A));
r = factor/rho; % Set a meaningful value for r
y = I - ( r * A );

S = inv( y ) - I; % see Gelfand's formula - convergence of the geometric series of matrices

%% 5) Estimating energy scores
if (verbose)
    fprintf('5) Estimating relevancy scores \n');
end
WEIGHT = sum( S , 2 ); % prob. scores s(i)

%% 6) Ranking features according to s
if (verbose)
    fprintf('6) Features ranking  \n');
end
[~ , RANKED ]= sort( WEIGHT , 'descend' );

RANKED = RANKED';
WEIGHT = WEIGHT';


end

%  =========================================================================
%   Reference   : Infinite Latent Feature Selection
%   Author      : Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli
%   Link        : preprint: https://arxiv.org/abs/1707.07538
%   ProjectPage : http://giorgioroffo.uk
%  =========================================================================

function [G] = LearningGraphWeights( train_x , train_y, TT, verbose )

if (verbose)
    fprintf(['\n+ PAMI - Feature selection: inf-VR \n' ...
        ' This procedure (PLSA-like) aims to discover something abount the meaning \n' ...
        ' behind the tokens, about more complex latent constructs in the\n' ...
        ' features distribution. Latent variables - factors - are combinations of observed\n' ...
        ' values (tokens) which can co-occur in different features.\n' ...
        ' Different from PLSA, the presence of a value in two or more feature\n' ...
        ' distributions cannot be subject to:\n' ...
        ' Polysemy: the same (observed) token have a unique meaning\n' ...
        '           (e.g., f(1) 1 -> good sample, f(2) 1 -> the same)\n' ...
        ' Or, \n' ...
        ' Synonymy: different tokens cannot have the same meaning:\n'...
        '           (e.g., f(1) 1 -> good sample, f(2) 2 -> bad sample)\n']);
end
if (nargin<3)
    verbose = 0;
end


% EM Parameters
params.iter = 15;

numFeat = size(train_x,2);
numSamp = size(train_x,1);

num_classes = length(unique(train_y));
unique_class_labels = sort(unique(train_y),'descend');
right_labels = 999*ones(size(train_y));

% rename labels: from 1 to num classes
for ii=1:num_classes
    right_labels(train_y == unique_class_labels(ii)) = ii;
end

for ii=1:num_classes
    
    c{ii,1} = train_x(right_labels==ii,:);
    mu_c(ii,:) = mean(c{ii});
    st(ii,:)   = std(c{ii}).^2;
    
end

st(st<1000000*eps)=1;

% Priors knowledge on data: Separation Scores for each dimension
% Some tricks to avoid numeric problems: NaN or Inf
% st_est(st_est<=0.00000001) = 1000;
% priors_sep_scores = (priors_sep_scores.^2) ./ st_est;
% priors_sep_scores = 1+(99*([priors_sep_scores-min(priors_sep_scores)]/max(priors_sep_scores-min(priors_sep_scores))));
% 
% % Check and remove unwanted values
% priors_sep_scores(isnan(priors_sep_scores)) = 1;
% priors_sep_scores(isinf(priors_sep_scores)) = 1;                
 
%% Latent Variables
numFactors = 2; % Num of Latent Variables - F1: Relevant Feature, F2: Not Relevant

% We define a "dictionary" or "vocabulary" as a row vector of 'numTokens'.
% Note, this first operation is a mapping from the raw features values to a
% first level analysis where the allowed tokes are:
% N: data is well represented (if the sample is closer to its correct class mean)
% 1: bad sample (otherwise)
token = [1:TT]; % the order of the tokens (multinominal vars) in the dictionary is fixed.
flag = 1;
numTokens = length(token);
% this dictionary can also be augmented with more symbols for intermediate positions.

% Given the dictionary, we consider each feature vector as a document,
% A feature (document) is a vector where each entry corresponds to a
% different token (term) and the number at that entry corresponds to how many times
% that token (term) was present in the feature distribution.
tokenFeatMatrix = zeros(numTokens,numFeat);
priors_sep_scores = zeros(1,numFeat);
% In order to create the token by feature matrix, we consider for each
% sample in the feature distribution its closest class mean, in order to
% map each value to a very simple representation.
for f=1:numFeat
    d = ( bsxfun( @minus, train_x(:,f), mu_c(:,f)' )).^2 / sum( st(:,f) );  % Multi-class fisher scores
    prob_class_est = (abs(d)./repmat(sum(abs(d),2),[1,size(d,2)]));
    prob_class = zeros(numSamp,1);
    for ss=1:numSamp
        prob_class(ss,1) = prob_class_est(ss,right_labels(ss));
    end
    if ~isnan(sum(prob_class)) && ~isinf(sum(prob_class))
        tokenFeatMatrix(:,f) = histc(prob_class,linspace(min(prob_class),max(prob_class),TT));
    end
    priors_sep_scores(f) = [sum(tokenFeatMatrix(TT:-1:(TT-round(TT*0.35)),f))/numSamp];  

end
priors_sep_scores = 100*priors_sep_scores;

% Check and remove unwanted values
priors_sep_scores(isnan(priors_sep_scores)) = 1;
priors_sep_scores(isinf(priors_sep_scores)) = 1;       

%% 1) Initialization conditional probabilities
% We manually set the initial conditional probabilities  p(token | factor)
% We want the Factor 1 models the probability that a feature is Relevant,
% for this reason we assign high probability to F1 for the token 1 which
% represent good samples.
prob_token_factor = [linspace(5000,1,numTokens)', linspace(1,5000,numTokens)'] ; %  p(token | factor): Factor 1 will represent the discriminative topic
prob_token_factor = prob_token_factor ./ sum(prob_token_factor, 1);

% Initialize  p(factor | feat)
% Since Factor 1 represent the discriminative topic, we set high scores of good features on
% F1 and high scores for bad features on F2
prob_factor_feat = zeros(numFactors,numFeat); % init
% F1 Relevant: high scores for discriminative features
prob_factor_feat(1,:) = priors_sep_scores ; 
% F1 Irrelevant: high scores for unwanted features
prob_factor_feat(2,:) = 100-prob_factor_feat(1,:); 
 
prob_factor_feat = prob_factor_feat ./ sum(prob_factor_feat, 1);


% Initialize also  p(token | feat)
prob_token_feat = zeros(numTokens, numFeat); % p(token | feat)
for z = 1:numFactors
    prob_token_feat = prob_token_feat + ...
        prob_token_factor(:, z) * prob_factor_feat(z, :);
end

prob_factor_token_feat = cell(numFactors, 1);   % p(factor | feat, token)
for z = 1 : numFactors
    prob_factor_token_feat{z} = zeros(numTokens, numFeat);
end


% The implementation of PLSA + EM algorithm is based on the code at:
% https://github.com/lizhangzhan/plsa
% https://github.com/lizhangzhan/plsa/blob/master/plsa.m

%% 2) Expectation-Maximization: maximum log-likelihood estimations
if (verbose)
    disp('Expectation-Maximization: maximum log-likelihood estimations.');
end
Lt = []; % log-likelihood

for ii = 1 : params.iter
    %disp('E-step');
    for z = 1:numFactors
        prob_factor_token_feat{z} = (prob_token_factor(:,z) * prob_factor_feat(z,:)) .* ...
            tokenFeatMatrix ./ prob_token_feat;
    end

    %disp('M-step');
    %disp('update p(z|d)')
    for z = 1:numFactors
        prob_token_factor(:,z) = sum(prob_factor_token_feat{z}, 2);
    end
    prob_token_factor = prob_token_factor ./ sum(prob_token_factor, 2);

    %disp('update p(w|z)')
    for z = 1:numFactors
        prob_factor_feat(z,:) = sum(prob_factor_token_feat{z}, 1);
        prob_factor_feat(z,:) = prob_factor_feat(z,:) / sum(prob_factor_feat(z,:));
    end

    % update p(d,w) and calculate likelihood
    prob_token_feat(:) = 0;
    for z = 1:numFactors
        prob_token_feat = prob_token_feat + prob_token_factor(:,z) * prob_factor_feat(z,:);
    end
    L = sum(sum(tokenFeatMatrix .* log(prob_token_feat)));

    Lt = [Lt; L];
    
    if (verbose)
        fprintf('likelihood: %f\n', L);
    end
        
    if (verbose)
        plot(Lt,'r','linewidth',2);
        grid on
        title 'EM - Maximum likelihood';
        drawnow
    end
    
    if ii > 1
        if Lt(end) - Lt(end-1) < 1e-6
            break;
        end
    end
end


% EM Completed.
if (verbose)
    fprintf('Optimized: %.2f \n',abs(Lt(end)-Lt(1)));
end
factor_representing_relevancy = 1;

% Building the graph:
% In order to connect features, we compute the joint probability
% P( factor_1 | feat_i, feat_j ) = P( factor_1 | feat_i ) * P( factor_1 | feat_j )
G = prob_factor_feat(factor_representing_relevancy,:)'*prob_factor_feat(factor_representing_relevancy,:);

end

%  ------------------------------------------------------------------------
%If you use our toolbox (or method included in it), please consider to cite:

%[1] Roffo, G., Melzi, S., Castellani, U. and Vinciarelli, A., 2017. Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach. arXiv preprint arXiv:1707.07538.

%[2] Roffo, G., Melzi, S. and Cristani, M., 2015. Infinite feature selection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 4202-4210).

%[3] Roffo, G. and Melzi, S., 2017, July. Ranking to learn: Feature ranking and selection via eigenvector centrality. In New Frontiers in Mining Complex Patterns: 5th International Workshop, NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016, Revised Selected Papers (Vol. 10312, p. 19). Springer.

%[4] Roffo, G., 2017. Ranking to Learn and Learning to Rank: On the Role of Ranking in Pattern Recognition Applications. arXiv preprint arXiv:1706.05933.
%  ------------------------------------------------------------------------

