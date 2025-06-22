# general_seedcracker
A CUDA Structure seed cracker that supports shipwrecks (1.16+), ruined portals, and villages (1.18+)


Compiled using:

nvcc -o general_seedcracker general_seedcracker.cu -std=c++17 -O3

Run using

./general_seedcracker {textfilename.txt}

Exmaple input:

-54, -14, COUNTERCLOCKWISE_90, sideways_fronthalf, Ocean

112, 89, CLOCKWISE_180, rightsideup_full_degraded, Beached

55, -9, CLOCKWISE_180, taiga_meeting_point_1, 3, no

52, 17, CLOCKWISE_180, portal_1, yes, 1


For villages the final constraint (yes or no) means if it is abandoned and the number before is biome type (1 = plains, 2 = snowy, 3 = taiga, 4 = savanna, 5 = desert)

For portals the second to last constraint (yes or no) means if it is mirrored and the final constraint is biome type which does affect portal tye (1 = most biomes, 2 = desert, nether (after 1.18), swamp (not mangrove though), ocean, 3 = jungle (any kind of portal that has vines growing on it)
