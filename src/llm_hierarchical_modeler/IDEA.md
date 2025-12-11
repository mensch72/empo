# Using LLMs to construct situational hierarchies of MDPs (or POSGs)


## Hierarchical Decision Context (HDC) JSON

The HDC encodes all variable data we need to know to perform the hierarchical decision algorithm.
All or parts of it can be passed to the LLM in prompts.

```
<HDC> = {
    "hierarchy": [ <level>, ..., <level> ],  # sorted from coarsest to finest
    "history": [ <observation|act>, ..., <observation|act> ],
}

    <level> = {
        "id": <string>,
        "model": [ <state_data>, ... ],
        "model_complete": <bool>,  # whether no states etc. are missing
        "model_up_to_date": <bool>,  # whether model has been updated on all observations
        "state_beliefs": <state_beliefs>,
        "?state_beliefs_status": "complete", "missing_probabilities", "missing_states"
        "?current_action_id": <string>
    }

        <state_data> = {
            "id": <string>,
            "desc": <string>,
            "terminal": <bool>,
            "?next_coarser_state_id": <string>  # if terminal
            "?active_agents": [ <agent_data>, ...],  # if not terminal
            "?active_agents_complete": <bool>,  # if not terminal
            "?transitions": [ <transition_data>, ...],  # if not terminal
            "?transitions_complete": <bool>  # if not terminal
        }

            <agent_data> = {
                "id": <string>,
                "type": "human" | "robot",
                "possible_actions": [ <action_data>, ... ],
                "possible_actions_complete": <bool>,
                "?habitual_probabilities": [ <float>, ... ],  # if human (= nu_h * pi^0_h in paper)
                "?others_believed_probabilities": [ <float>, ... ],  # if human. what other humans assume this human's policy is
                "?inverse_temp": <float>,  # if human (= beta_h in paper)
                "?discount_factor": <float>,  # if human (= gamma_h in paper)
            }

                <action_data> = {
                    "id": <string>,
                    "desc": <string>
                }
            
            <transition_data> = {
                "action_pattern": <action_pattern>,
                "possible_successors": [ <state_id>, ... ],
                "possible_successors_complete": <bool>,
                "?probabilities": [ <float>, ... ]
            }

                <action_pattern> = ?  # some structured representation of any proposition about what agents took what actions

        <state_beliefs> = {
            "possible_states": [ <state_id>, ... ],
            "possible_states_complete": <bool>,
            "?probabilities": [ <float>, ... ]
        }

    <observation> = {
        "type": "observation",
        "content": <JSON>
    }

    <act> = {
        "level_id" = <string>,
        "action_id" = <string>
    }
```

## LLM tasks

- form an initial model:
    - decide level of granularity
    - assemble list of possible initial states (without agent/transition data at first)
    - recursively complete state data until no further possible successors found

- update the model after observation:
    - decide if current state might not be covered by possible states in model. if so: --> add current state's data

- add/complete a state's data:
    - desc, terminal, next_coarser_state_id
    - successively compile list of active agents (without action data at first)
    - successively complete each agent's data
    - successively compile transition data

- complete an agent's data:
    - type
    - successively compile list of actions
    - at once create habitual_probabilities and others_believed_probabilities





# [OLD]


## glossary

- level $l$ = level of the model hierarchy. coarser levels have smaller numbers, beginning with 1. 
- state $s$ = description of the state of the world at some level of granularity, *regardless* of what part of it can be observed by who
- hierarchical state-action context $C_l$ = state $s_l$ + attempted action $a_l$ + possible successor states $s'_l$ (but no transition probabilities!)

## LLM tasks

### given state, imagine actions

- given: 
    - coarser level hierarchical state-action contexts $C_1$ ... $C_{l-1}$
    - current level state $s_l$
    - possibly incomplete list $A_l$ of possible actions $a_l$ that one might plausibly attempt to implement the coarser-level action $a_{l-1}, without any mention of possible consequences yet, but each with:
        - a short id, a human-readable label, a short description, a detailed description
        - a justification why it is plausible that one might attempt it
        - a justification why it is disjoint from the other actions
- sought: 
    - an assessment whether the list $A_l$ is already complete
    - if not: one additional action $a_l$ for that list $A_l$

#### general JSON schema for this task

```
{
    "hierarchical state-action context": {
        "level 1 (coarsest)": {
            "state": {
                "id": "1",
                "label": "53, tenured, work on AI safety",
                "short description": "Agent is 53 years old, a tenured scientist, and working on AI safety",
                "detailed description": "..."
            },
            "attempted action": {
                "id": "a",
                "label": "maximize human power",
                "short description": "Agent attempts to develop metrics and AI algorithms for maximizing total human power",
                "detailed description": "..."
            },
            "possible next states": [
                { "id": ..., ... }, # as above
                ...
            ]
        },
        "level 2 (finer)": {
            "state": { ... },
            "attempted action": { ... },
            "possible next states": [ ... ]
            # as above
        },
        ...,
        "level 5 (current)": {
            "state": { ... }
            # no attempted action or possible next states at this current level, because that is what we are about to construct!
        }
    },
    "possible distinct actions to attempt (incomplete?)": [
        { 
            "id": ..., "label": ..., "short description": ..., "detailed description": ...,
            "why plausible?": ...,
            "why distinct from other actions?": ...
        },
        ... # more such entries
    ]
}
```

#### prompt

You are helping develop a hierarchy of MDPs, each level refining one action from the previous level. 
You will propose one additional action that the agent might attempt in a certain context.
I will give you a JSON describing the current context, which is a list that species the current state and attempted action at the coarsest model level and at the finer model levels up to the current model level's current state.
It also contains a possibly incomplete list of possible actions at the current model level's current state. 
Here's the context:

```
    {
        "hierarchical state-action context": {
            "level 1 (coarsest)": {
                "state": {
                    "id": "1",
                    "label": "53, tenured, work on AI safety",
                    "short description": "Agent is 53 years old, a tenured scientist, and working on AI safety",
                    "detailed description": "..."
                },
                "attempted action": {
                    "id": "a",
                    "label": "maximize human power",
                    "short description": "Agent attempts to develop metrics and AI algorithms for maximizing total human power",
                    "detailed description": "..."
                },
                "possible next states": [
                    { "id": "2", "label": "55, fired, looking for a job" },
                    { "id": "3", "label": "55, solved AI safety, hired by major lab" },
                    { "id": "4", "label": "55, solved AI safety, ignored by industry" }
                ]
            },
            "level 2 (finer)": {
                "state": {
                    "id": "1",
                    "label": "has some ideas on how to maximize human power",
                    "short description": "Agent has some initial ideas on how to maximize human power"
                },
                "attempted action": {
                    "id": "a",
                    "label": "develop metrics for human power",
                    "short description": "Agent attempts to develop metrics for measuring human power"
                },
                "possible next states": [
                    { "id": "2", "label": "has a working metric for human power" },
                    { "id": "3", "label": "has a theoretical framework for human power" },
                    { "id": "4", "label": "has no useful metric or framework" }
                ]
            },
            "level 3 (current)": {
                "state": { 
                    "id": "1",
                    "label": "has an initial idea for a metric"
                }
            }
        },
        "possible distinct actions to attempt (incomplete?)": [
            { 
                "id": "a", "label": "formalize idea", "short description": "start working on the idea theoretically", "detailed description": "...",
                "why plausible?": "Formalization is a common first step in scientific work, especially for theoretical ideas.",
                "why distinct from other actions?": "This action focuses on the theoretical aspect of the idea, distinguishing it from practical implementations or empirical testing."
            },
            { 
                "id": "b", "label": "test idea with a small experiment", "short description": "conduct a small experiment to test the idea", "detailed description": "...",
                "why plausible?": "Testing ideas with experiments is a standard scientific method to validate hypotheses.",
                "why distinct from other actions?": "This action emphasizes empirical validation, which is different from theoretical work."
            },
            { 
                "id": "c", "label": "discuss idea with colleagues", "short description": "get feedback from peers on the idea", "detailed description": "...",
                "why plausible?": "Collaboration and peer feedback are essential in scientific research.",
                "why distinct from other actions?": "This action focuses on social interaction and feedback, rather than direct work on the idea."
            }
        ]
    }
```

You will now please return a JSON specifying either (i) one more possible entry for the list of possible distinct actions to attempt, or (ii) a justification why the list is complete.

Please return only a JSON in the following format, and nothing else. Either return

```
    { 
        "id": "the next free single letter", 
        "label": "a short human readable label", 
        "short description": "a short summary of what the action does", 
        "detailed description": "a detailed description of what it means to attempt this action",
        "why plausible?": "a justification why this action is a plausible thing to try in order to implement the given next-coarser-level action",
        "why distinct from other actions?": "a justification why this action is different from all other actions in the list"
    }
```

or return

```
    {
        "why complete?": "a justfication why you consider the list of possible actions is already complete"
    }
```

### given action, imagine successor states

- given:
    - coarser level action contexts
    - current level state and action
    - possibly incomplete list of possible successor states that might plausibly result from attempting this action, without probabilities yet (? or with initial probability assessments ?), but each with:
        - a short id, a human-readable label, a short description, 
        - a longer description
        - a flag whether this implements one of the next-coarser-level successor states. if so: 
            - which state
            - a justtification for that assessment 
        - a justification why it is plausible that this might result
        - a justification why it is disjoint from the other successor states
- sought: 
    - an assessment whether the list is complete
    - if not: one additional state for that list

#### prompt

You are helping develop a hierarchy of MDPs, each level refining one action from the previous level. 
You will propose one additional possible consequence that might result from a certain action the agent can attempt in a certain context.
I will give you a JSON describing the current context, which is a list that species the current state and attempted action at the coarsest model level and at the finer model levels up to the current model level's current state and attempted action.
It also contains an incomplete list of possible successor states resulting from that action at the current model level. 
Here's the context:

    {
        "hierarchical state-action context": {
            "level 1 (coarsest)": {
                "state": {
                    "id": "1",
                    "label": "53, tenured, work on AI safety",
                    "short description": "Agent is 53 years old, a tenured scientist, and working on AI safety",
                    "detailed description": "..."
                },
                "attempted action": {
                    "id": "a",
                    "label": "maximize human power",
                    "short description": "Agent attempts to develop metrics and AI algorithms for maximizing total human power",
                    "detailed description": "..."
                },
                "possible next states": [
                    { "id": "2", "label": "55, fired, looking for a job" },
                    { "id": "3", "label": "55, solved AI safety, hired by major lab" },
                    { "id": "4", "label": "55, solved AI safety, ignored by industry" }
                ]
            },
            "level 2 (finer)": {
                "state": {
                    "id": "1",
                    "label": "has some ideas on how to maximize human power",
                    "short description": "Agent has some initial ideas on how to maximize human power"
                },
                "attempted action": {
                    "id": "a",
                    "label": "develop metrics for human power",
                    "short description": "Agent attempts to develop metrics for measuring human power"
                },
                "possible next states": [
                    { "id": "2", "label": "has a working metric for human power" },
                    { "id": "3", "label": "has a theoretical framework for human power" },
                    { "id": "4", "label": "has no useful metric or framework" }
                ]
            },
            "level 3 (current)": {
                "state": { 
                    "id": "1",
                    "label": "has an initial idea for a metric"
                },
                "attempted action": {
                    "id": "a", 
                    "label": "formalize idea", 
                    "short description": "start working on the idea theoretically", 
                    "detailed description": "...",
                },
            }
        },
        "possible distinct successor states (incomplete?)": [
            { 
                "id": "1",
                "label": "formalization failed",
                "short description": "the agent did not succeed in finding a convincing formalization after a month of work",
                "detailed description": "...",
                "corresponding coarser-level state": null,
                "why plausible?": "one can always fail",
                "why distinct from other actions?": "this is the first entry in the list...",
                "why this mapping to coarser level?": "this is not the end of develop metrics for human power, one can try other things."
            }
        ],
        "successor list complete?": {
            "value": false,
            "justification": "there's certainly other things that might happen..."
        }
    }

You will now please return a JSON specifying one more possible entry for the list of possible distinct successor states.
Please return only a JSON in the following format, and nothing else:

    { 
        "id": "a consecutive integer", 
        "label": "a short human readable label", 
        "short description": "a short summary of what defines that state", 
        "detailed description": "a detailed description of what it means to be in that state",
        "corresponding coarser-level state": null | "the id of that state in the previous model level which this state is a special case of, if any, otherwise null",
        "why plausible?": "a justification why this state is a plausible next state that could result from the attempted action in the current state",
        "why distinct from other actions?": "a justification why this state is different from all other states in the list",
        "why this mapping to coarser level?": "a justification why this state maps to the specified coarser-level state or does not map to any coarser-level states"
    }

### given successor state list, estimate probabilities

TODO

### given observation, determine state and assess need for updating model

TODO

### given observation, update model

TODO
