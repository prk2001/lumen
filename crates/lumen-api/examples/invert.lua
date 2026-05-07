-- Example Lumen Lua plugin: simple per-channel RGB inversion.
--
-- Loaded by `lumen_api::load_lua_plugin`. The plugin host calls
-- `effect_metadata()` once at load time and `apply(width, height, pixels,
-- params)` for every frame. `pixels` is a flat Lua table of f32s in
-- interleaved RGBA order (length = width * height * 4).

function effect_metadata()
  return {
    id = "plugin.invert",
    display_name = "Invert",
    description = "Inverts RGB channels (alpha untouched).",
    category = "Color",
    version = 1,
    params = {
      {
        id = "amount",
        display_name = "Amount",
        description = "0.0 = pass-through, 1.0 = fully inverted.",
        kind = "float",
        default = 1.0,
        min = 0.0,
        max = 1.0,
      },
    },
  }
end

function apply(width, height, pixels, params)
  local amount = params.amount or 1.0
  local inv = 1.0 - amount
  for i = 1, #pixels, 4 do
    pixels[i]     = (1.0 - pixels[i])     * amount + pixels[i]     * inv
    pixels[i + 1] = (1.0 - pixels[i + 1]) * amount + pixels[i + 1] * inv
    pixels[i + 2] = (1.0 - pixels[i + 2]) * amount + pixels[i + 2] * inv
    -- pixels[i + 3] = alpha, left untouched.
  end
  return pixels
end
