# RoboPRO Benchmark Tasks & Data Collection Plan

## Data Collection Target

Each task should have:
- **Clean**: 100 episodes (obstacle_density = 0)
- **Cluttered**: 10 densities (d6 through d15) x 10 episodes each = 100 episodes total

Total per task: 200 episodes (100 clean + 100 cluttered)

**Total: 80 tasks across 4 domains** (Study, Office, KitchenL, KitchenS) = 16,000 episodes target.

---

## Study (20 tasks)

| # | Task Description | Code File | Type |
|---|---|---|---|
| 1 | Move the book onto the table | `move_book_onto_table` | Atomic |
| 2 | Move the cup to the right/left | `move_cup` | Atomic |
| 3 | Move the cup next to the book | `move_cup_next_to_book` | Atomic |
| 4 | Move the cup onto the table | `move_cup_onto_table` | Atomic |
| 5 | Move the pen to the box | `move_pen_to_box` | Atomic |
| 6 | Move the seal next to the box | `move_seal_next_to_box` | Atomic |
| 7 | Move the seal next to the pencup | `move_seal_next_to_pencup` | Atomic |
| 8 | Move the seal onto the table | `move_seal_onto_table` | Atomic |
| 9 | Put the cup in the box | `put_cup_in_box` | Atomic |
| 10 | Put the cup on the coaster | `put_cup_on_coaster` | Atomic |
| 11 | Put the cup on the table | `put_cup_on_table` | Atomic |
| 12 | Put the glue in the box | `put_glue_in_box` | Atomic |
| 13 | Put the pen in the box | `put_pen_in_box` | Atomic |
| 14 | Put the pen in the pencup | `put_pen_in_pencup` | Atomic |
| 15 | Put the seal in the box | `put_seal_in_box` | Atomic |
| 16 | Empty the box | `empty_box` | Compositional |
| 17 | Move the cup, then put the pen in the cup | `move_cup_put_pen_in_cup` | Compositional |
| 18 | Move the cups into the box | `move_cups_into_box` | Compositional |
| 19 | Move the seal next to box, put cup in box | `move_seal_cup_next_to_box` | Compositional |
| 20 | Move the seal onto the book | `move_seal_onto_book` | Compositional |

## Office (20 tasks)

| # | Task Description | Code File | Type |
|---|---|---|---|
| 1 | Move the milktea to the right of the laptop | `put_milktea_next_to_laptop` | Atomic |
| 2 | Put the stapler in the open drawer | `put_stapler_in_drawer` | Atomic |
| 3 | Stack the book from file holder on top of the book on the table | `put_book_on_book` | Atomic |
| 4 | Put the phone on the holder | `put_phone_on_holder` | Atomic |
| 5 | Move the phone to beside the cube | `put_phone_next_to_cube` | Atomic |
| 6 | Put the mouse on the mousepad | `put_mouse_on_pad` | Atomic |
| 7 | Place the milktea on the lower shelf | `put_milktea_on_shelf` | Atomic |
| 8 | Pick the mouse from the open drawer and place it to the right of the stapler | `put_mouse_next_to_stapler` | Atomic |
| 9 | Open the drawer | `open_drawer` | Atomic |
| 10 | Close the drawer | `close_drawer` | Atomic |
| 11 | Move the book from the shelf to the lower level of the file holder | `put_book_in_fileholder` | Atomic |
| 12 | Place the rubiks cube to the left of the milktea | `put_rubikscube_next_to_milktea` | Atomic |
| 13 | Place the stapler next to the mouse | `put_stapler_next_to_mouse` | Atomic |
| 14 | Move rubiks cube to drawer | `put_rubikscube_in_drawer` | Atomic |
| 15 | Place the stapler on the book | `put_stapler_on_book` | Atomic |
| 16 | Open the drawer, place the stapler inside, close the drawer | `store_stapler_in_drawer` | Compositional |
| 17 | Milktea to coaster, rubiks cube to shelf, notebook to book | `organize_table` | Compositional |
| 18 | Mouse to pad, phone to holder, milktea to shelf | `set_up_table` | Compositional |
| 19 | Open drawer, rubiks cube to shelf, close drawer | `store_rubikscube_on_shelf` | Compositional |
| 20 | Mouse from drawer to pad, close drawer, book to file holder | `move_items_around` | Compositional |

## KitchenL (20 tasks)

| # | Task Description | Code File | Type |
|---|---|---|---|
| 1 | Move the bottle | `move_bottle` | Atomic |
| 2 | Pick up the bottle from the fridge | `pick_bottle_from_fridge` | Atomic |
| 3 | Pick up the box drink from the basket | `pick_boxdrink_from_basket` | Atomic |
| 4 | Pick up the can from the basket | `pick_can_from_basket` | Atomic |
| 5 | Pick up the can from the cabinet | `pick_can_from_cabinet` | Atomic |
| 6 | Pick up the milk box from the fridge | `pick_milk_box_from_fridge` | Atomic |
| 7 | Pick up the sauce can from the cabinet | `pick_sauce_can_from_cabinet` | Atomic |
| 8 | Put the bottle in the basket | `put_bottle_in_basket` | Atomic |
| 9 | Put the bottle in the fridge | `put_bottle_in_fridge` | Atomic |
| 10 | Put the can in the cabinet | `put_can_in_cabinet` | Atomic |
| 11 | Put the can in front of the microwave | `put_can_infront_of_microwave` | Atomic |
| 12 | Put the can next to the basket | `put_can_next_to_basket` | Atomic |
| 13 | Put the milk box in the fridge | `put_milk_box_in_fridge` | Atomic |
| 14 | Put the sauce can in the basket | `put_sauce_can_in_basket` | Atomic |
| 15 | Put the sauce can in the cabinet | `put_sauce_can_in_cabinet` | Atomic |
| 16 | Move the bottle from the fridge next to the can | `move_bottle_from_fridge_next_to_can` | Compositional |
| 17 | Move the can from the cabinet to the basket | `move_can_from_cabinet_to_basket` | Compositional |
| 18 | Move the milk and close the fridge | `move_milk_close_fridge` | Compositional |
| 19 | Put the can away and close the cabinet | `put_can_close_cabinet` | Compositional |
| 20 | Swap the can with the bottle in the basket | `switch_can_with_bottle_in_basket` | Compositional |

## KitchenS (20 tasks)

| # | Task Description | Code File | Type |
|---|---|---|---|
| 1 | Close the microwave door | `close_microwave_ks` | Atomic |
| 2 | Drop the apple in the bin | `drop_apple_in_bin_ks` | Atomic |
| 3 | Move the hamburger onto the plate | `move_hamburger_onto_plate_ks` | Atomic |
| 4 | Pick up the apple from the bowl | `pick_apple_from_bowl_ks` | Atomic |
| 5 | Pick up the apple from the sink | `pick_apple_from_sink_ks` | Atomic |
| 6 | Pick up the fork from the sink | `pick_fork_from_sink_ks` | Atomic |
| 7 | Pick up the hamburger from the microwave | `pick_hamburger_from_microwave_ks` | Atomic |
| 8 | Place the bowl in the dishrack | `place_bowl_in_dishrack_ks` | Atomic |
| 9 | Put the bowl in the sink | `put_bowl_in_sink_ks` | Atomic |
| 10 | Put the bread on the board | `put_bread_on_board_ks` | Atomic |
| 11 | Put the hamburger in the microwave | `put_hamburger_in_microwave_ks` | Atomic |
| 12 | Put the plate in the sink | `put_plate_in_sink_ks` | Atomic |
| 13 | Put the spoon in the dishrack | `put_spoon_in_dishrack_ks` | Atomic |
| 14 | Put the spoon in the sink | `put_spoon_in_sink_ks` | Atomic |
| 15 | Put the spoon on the plate | `put_spoon_on_plate_ks` | Atomic |
| 16 | Put the apple in the bin, the bowl in the dishrack, and the spoon in the sink | `chain_apple_bin_bowl_rack_spoon_sink_ks` | Compositional |
| 17 | Put the apple from the sink on the plate and the bread on the board | `chain_apple_sink_plate_bread_board_ks` | Compositional |
| 18 | Move the bowl from the sink to the dishrack and the apple into the sink | `chain_bowl_rack_apple_sink_ks` | Compositional |
| 19 | Put the hamburger in the microwave and close the door | `chain_heat_hamburger_ks` | Compositional |
| 20 | Take the hamburger out of the microwave, place it on the bowl, and close the door | `chain_serve_hamburger_ks` | Compositional |

## HuggingFace Dataset

Repository: TODO: HuggingFace dataset

Layout (same shape for all four domains — `study/`, `office/`, `kitchenl/`, `kitchens/`):

```
{domain}/
  {task_name}/
    clean/    # 100 episodes, density 0
    d6/       # 10 episodes, density 6
    ...
    d15/      # 10 episodes, density 15
```
