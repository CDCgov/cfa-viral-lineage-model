# Architecture

- Raw data comes in whatever form
- They get cleaned up to some common schema
- A "model" is everything between the common-schema input and some common-schema output
- There might be further post-processing beyond that, that is still upstream of visualizations and so forth
- An "evaluator" is some thing that coordinates the action of multiple score functions

```mermaid
flowchart TD

%% Raw data and common pre-processing
raw_data[/Raw data/] --> preproc[Common pre-processing]
preproc --> clean_training[/Clean training data/]
preproc --> clean_eval[/Clean eval data/]

%% Individual models
subgraph model1[Model 1]
  preproc1[Model-specific pre-processing]
  comp1[Mathematical modeling]
  postproc1[Model-specific post-processing]
end

subgraph model2[Model 2]
  preproc2[Model-specific pre-processing]
  comp2[Mathematical modeling]
  postproc2[Model-specific post-processing]
end

output1[/Model 1 output/]
output2[/Model 2 output/]

%% Flow through the models
clean_training --> preproc1 --> comp1 --> postproc1 --> output1
clean_training --> preproc2 --> comp2 --> postproc2 --> output2

%% Evaluation
subgraph Evaluator
    score1[Score function 1]
    score2[Score function 2]
end

outputs --> score1
outputs --> score2
clean_eval --> score1
clean_eval --> score2
score1 --> scores
score2 --> scores
scores[/Scores/]

%% Common post-processing
postproc[Common post-processing] --> outputs[/Clean outputs/]

%% Diagnostics
outputs --> dx_scripts[Diagnostic scripts] --> dx[/Diagnostic files & figures/]
output1 --> postproc
output2 --> postproc

%% Visualizations
viz_scripts[Visualization scripts]
outputs --> viz_scripts
scores --> viz_scripts
```
