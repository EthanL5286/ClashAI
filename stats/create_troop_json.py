import json

card_name = input("card name ")
hp = input("hp ")
direct_damage = input("direct damage ")
splash_damage = input("splash damage ")
hit_speed = input("hit speed ")
speed = input("speed ")
melee_range = input("melee range ")
range = input("range ")
targets = input("targets ")
flying = input("flying ")
spell = input("spell ")
spell_radius = input("spell radius ")
# troop_count = input("troop count ")
# elixir = input("elixir ")

stats_file = open("troop_stats.json", "r")
card_json = json.load(stats_file)
stats_file.close()

card_json[card_name] = {
    "hp" : hp,
    "direct_damage" : direct_damage,
    "splash_damage" : splash_damage,
    "hit_speed" : hit_speed,
    "speed" : speed,
    "melee_range" : melee_range,
    "range" : range,
    "targets" : targets,
    "flying" : flying,
    "spell" : spell,
    "spell_radius" : spell_radius
    # "troop_count" : troop_count,
    # "elixir" : elixir
}


stats_file = open("troop_stats.json", "w")
stats_file.write(json.dumps(card_json, indent=4))
stats_file.close()
