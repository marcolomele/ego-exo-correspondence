# Title
Super-human vision for object localisation: enhancing egocentric views with exocentric priors in occluded environments. 

# Problem Formulation
## Two line
Ego is detailed, describes action, and drives decisions; exo is global, covers the environment, and offers different angles. Combined, they give super-human vision. 

We will focus on enhancing object localisation under occlusion. 

## Context
The egocentric POV is one of the 5 senses, i.e. sources of information, that our brain evolved to interact with the real world. If we aim at building artificial intelligence by mimicking human intelligence, than this ego feels the most "natural" approach to computer vision: teach machine learning models how to see the world they way we see it with our eyes. 

Egocentric view data (Ego) is video taken from the point of view (POV) of humans. It is highly detailed about present state, and carries unique information like hand gestures.

However, ego is constrained to where our bodies can physically take us and focuses on a narrow portion of the available environment canvas. This means that global information about the environment and elements/events in it is missed. 

Exocentric view (exo) is vide taked from any other POV than the human's, typically cameras on walls, stands, or drones. It offers a global view of the environment where the action is talking place, from multiple angles, thus covering multiple portions at the same time. 

However exo lacks detail and dexterity of the action that human is focused on. 

Therefore, we need to combine ego and exo views to benefit of their complementarity and to produce valuable solutions. 

Combining ego and exo means enhancing human vision with information from the environment. This has several downstream applications, from efficient space exploration to object detection. 

In this project, we will focus on the latter problem: object localisation under occlusion.

## Importance and Relevance of the Problem
Today, AR technology give us glipse of the future of mobile computing. These are tools designed to interact with our surrounding physical environment and give even more relevant for various tasks.

A common task when doing manual jobs is searching for a specific object in a room. Being able to localize objects even when they are partially or fully occluded is crucial for robust perception — for both humans and machines. 

Accurate localization under occlusion powers a wide range of real-world applications: assistive AR for low-vision users, factory and warehouse robotics navigating cluttered shelves, maintenance and repair guidance where tools are hidden behind equipment, and search-and-rescue or firefighting systems that need to highlight objects or hazards through smoke and obstacles.

## Data Sourcing Strategy
Leverage Meta's Ego-Exo4D dataset. It contains more than 1000+ hours of multimodal multiview videos captured simultaneouslu and timesynced between egocentric and exocentric views of humans doing skilled activities, including cooking, instrument playing, dancing, and soccer. 

The authors of the database propose several pre-defined tasks. We focus on the “Correspondence” split, which conveniently already gives us synchronized frame pairs:
* Ego video (head-mounted, first-person view)
* Exo video (fixed or handheld third-person cameras)
* Per-frame masks marking the same object in both views
* Visibility flags (tells us if the object is occluded in ego / visible in exo)
* Camera calibration JSONs (intrinsics + extrinsics) – for analysis if ML model is able to learn Geometric mapping without coordinate inputs. 

## Proposed Solution (High-Level Overview): Provide a overview of your proposed approach or solution. Focus on the key idea and overall strategy rather than implementation details.

to add:
Exo->Ego correspondence
Input:
Exocentric Video: RGB video sequence.
Egocentric Video: RGB video sequence.
Exocentric query track: Framewise object binary masks
Output:
Egocentric output masks: Framewise object binary masks

## Performance Evaluation Approach: Explain how you plan to assess your solution’s effectiveness. Specify the metrics, benchmarks, or evaluation criteria you intend to use and why they are appropriate for your problem.
