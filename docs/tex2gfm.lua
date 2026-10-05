-- tex2gfm.lua
-- Pandoc Lua filter used to generate README.md from FMC_manual.tex
--   pandoc FMC_manual.tex -f latex -t gfm --lua-filter=tex2gfm.lua --citeproc ...
--
-- LaTeX numbers sections and figures automatically and resolves \ref; pandoc does not.
-- This filter reproduces that numbering (article class: sections continue across parts;
-- figures are numbered consecutively) and writes the GitHub conventions:
--   * headings with their number ("### 5. Input", "#### 5.1 Output examples")
--   * "*Figure 8.2. caption*" below every figure
--   * \ref -> section/figure numbers (sections link to their heading anchor)
--   * maths with the GitHub delimiters  $`...`$  and  ```math
--   * description lists as bullet lists with the term in bold
--   * the table of contents and the reference list; the unnumbered sections that precede the
--     first part in the PDF (Acknowledgments) are moved to the end of the README

local stringify = pandoc.utils.stringify

local function roman(n)
  local r = { 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X' }
  return r[n] or tostring(n)
end

-- GitHub anchor: lower case, keep letters, digits, spaces, hyphens and underscores; spaces -> hyphens
local function slug(s)
  s = s:lower()
  s = s:gsub("\u{2014}", ""):gsub("\u{2013}", "")
  s = s:gsub("[^%w%s%-_\u{80}-\u{10FFFF}]", "")
  s = s:gsub("%s", "-")
  return s
end

local function is_unnumbered(h)
  return h.classes:includes('unnumbered')
end

function Pandoc(doc)
  ------------------------------------------------------------------ pass 1: numbering
  local part, sec, sub, fig = 0, 0, 0, 0
  local label = {}          -- LaTeX label -> displayed number
  local anchor = {}         -- LaTeX label -> GitHub anchor (sections only)
  local toc = {}            -- entries for the table of contents
  local toc_moved = {}
  local blocks, moved = {}, {}
  local target = blocks
  for _, b in ipairs(doc.blocks) do
    if b.t == 'Div' and b.classes:includes('flushleft') then
      -- front-page license note: only for the PDF
    elseif b.t == 'Header' then
      local title = stringify(b.content)
      if b.level == 1 then
        part = part + 1
        target = blocks
        local text = 'Part ' .. roman(part) .. ' \u{2014} ' .. title
        table.insert(toc, { level = 1, text = text, anchor = slug(text) })
      elseif b.level == 3 then
        if is_unnumbered(b) then
          if part == 0 then
            target = moved
            table.insert(toc_moved, { level = 0, text = title, anchor = slug(title) })
          else
            target = blocks
            table.insert(toc, { level = 0, text = title, anchor = slug(title) })
          end
        else
          target = blocks
          sec = sec + 1; sub = 0
          local text = sec .. '. ' .. title
          label[b.identifier] = tostring(sec)
          anchor[b.identifier] = slug(text)
          table.insert(toc, { level = 2, text = text, anchor = slug(text) })
        end
      elseif b.level == 4 then
        sub = sub + 1
        label[b.identifier] = sec .. '.' .. sub
      end
      table.insert(target, b)
    elseif b.t == 'Figure' then
      fig = fig + 1
      label[b.identifier] = tostring(fig)
      table.insert(target, b)
    else
      table.insert(target, b)
    end
  end
  for _, e in ipairs(toc_moved) do table.insert(toc, e) end
  table.insert(toc, { level = 0, text = 'References', anchor = 'references' })

  ------------------------------------------------------------------ pass 2: rewrite
  local function inline_math(x)
    return pandoc.walk_block(x, { Math = function(m) return pandoc.Math('InlineMath', m.text) end })
  end

  local function dl_to_bullets(dl)
    local items = {}
    for _, item in ipairs(dl.content) do
      local term, defs = item[1], item[2]
      local blks = pandoc.List({})
      local first = true
      for _, def in ipairs(defs) do
        for _, bl in ipairs(def) do
          if first and (bl.t == 'Para' or bl.t == 'Plain') then
            local inl = pandoc.List({ pandoc.Strong(term), pandoc.Str(':'), pandoc.Space() })
            inl:extend(bl.content)
            blks:insert(pandoc.Plain(inl))
          else
            if first then blks:insert(pandoc.Plain({ pandoc.Strong(term), pandoc.Str(':') })) end
            blks:insert(bl)
          end
          first = false
        end
      end
      if first then blks:insert(pandoc.Plain({ pandoc.Strong(term) })) end
      table.insert(items, inline_math(pandoc.Div(blks)).content)
    end
    return pandoc.BulletList(items)
  end

  local function fix_inlines(blk)
    blk = pandoc.walk_block(pandoc.Div({ blk }), { DefinitionList = dl_to_bullets }).content[1]
    return pandoc.walk_block(pandoc.Div({ blk }), {
      Link = function(l)
        if l.attributes['reference-type'] == 'ref' then
          local key = l.attributes['reference']
          local num = label[key]
          if num then
            if anchor[key] then
              return pandoc.Link({ pandoc.Str(num) }, '#' .. anchor[key])
            end
            return pandoc.Str(num)
          end
        end
      end,
      Math = function(m)
        if m.mathtype == 'InlineMath' then
          return pandoc.RawInline('markdown', '$`' .. m.text .. '`$')
        else
          return pandoc.RawInline('markdown', '\n```math\n' .. m.text:gsub('^%s+', ''):gsub('%s+$', '') .. '\n```\n')
        end
      end,
      Str = function(s)
        if s.text:find('\u{00A0}') then
          return pandoc.Str((s.text:gsub('\u{00A0}', ' ')))
        end
      end,
      CodeBlock = function(c)
        if #c.classes == 0 then
          if c.text:sub(1, 1) == '@' then c.classes = { 'bibtex' } else c.classes = { 'text' } end
          return c
        end
      end,
      Table = function(t)
        -- header cells are written in bold in the LaTeX source; GitHub already renders them in bold
        local function unbold(cell)
          cell.contents = pandoc.walk_block(pandoc.Div(cell.contents), {
            Strong = function(s) return s.content end }).content
        end
        for _, row in ipairs(t.head.rows) do
          for _, cell in ipairs(row.cells) do unbold(cell) end
        end
        return t
      end,
    }).content[1]
  end

  local out = {}
  local fig_n, sec_n, sub_n, part_n = 0, 0, 0, 0
  local blocks_all = blocks
  local toc_done = false

  local function emit_toc()
    local items = {}
    local cur = nil
    local function item(e)
      return pandoc.Plain({ pandoc.Link({ pandoc.Str(e.text) }, '#' .. e.anchor) })
    end
    for _, e in ipairs(toc) do
      if e.level == 1 or e.level == 0 then
        cur = { item(e) }
        table.insert(items, cur)
      elseif e.level == 2 and cur then
        local sublist = cur[2]
        if not sublist then
          sublist = pandoc.BulletList({})
          cur[2] = sublist
        end
        table.insert(sublist.content, { item(e) })
      end
    end
    table.insert(out, pandoc.Header(2, { pandoc.Str('Contents') }))
    table.insert(out, pandoc.BulletList(items))
  end

  for _, b in ipairs(blocks) do
    if b.t == 'Header' then
      if not toc_done then emit_toc(); toc_done = true end
      local title = b.content
      if b.level == 1 then
        part_n = part_n + 1
        table.insert(out, pandoc.Header(2, { pandoc.Str('Part ' .. roman(part_n) .. ' \u{2014} ' .. stringify(title)) }))
      elseif b.level == 3 then
        if is_unnumbered(b) then
          table.insert(out, pandoc.Header(2, title))
        else
          sec_n = sec_n + 1; sub_n = 0
          local inl = pandoc.List({ pandoc.Str(sec_n .. '.'), pandoc.Space() }); inl:extend(title)
          table.insert(out, pandoc.Header(3, inl))
        end
      elseif b.level == 4 then
        sub_n = sub_n + 1
        local inl = pandoc.List({ pandoc.Str(sec_n .. '.' .. sub_n), pandoc.Space() }); inl:extend(title)
        table.insert(out, pandoc.Header(4, inl))
      elseif b.level == 5 then
        table.insert(out, pandoc.Header(5, title))
      elseif b.level == 6 then
        table.insert(out, pandoc.Para({ pandoc.Strong(title) }))
      else
        table.insert(out, b)
      end
    elseif b.t == 'Figure' then
      fig_n = fig_n + 1
      local num = tostring(fig_n)
      local img = nil
      pandoc.walk_block(b, { Image = function(i) img = i end })
      local src = img.src
      if not src:match('^docs/') then src = 'docs/' .. src end
      local cap = pandoc.List({ pandoc.Str('Figure ' .. num .. '.'), pandoc.Space() })
      if b.caption and b.caption.long and #b.caption.long > 0 then
        cap:extend(b.caption.long[1].content)
      end
      table.insert(out, pandoc.Para({ pandoc.Image({ pandoc.Str('Figure ' .. num) }, src) }))
      table.insert(out, fix_inlines(pandoc.Para({ pandoc.Emph(cap) })))
    else
      table.insert(out, fix_inlines(b))
    end
  end

  -- sections moved to the end (Acknowledgments)
  for _, b in ipairs(moved) do
    if b.t == 'Header' then
      table.insert(out, pandoc.Header(2, b.content))
    else
      table.insert(out, fix_inlines(b))
    end
  end

  -- reference list (filled by --citeproc)
  table.insert(out, pandoc.Header(2, { pandoc.Str('References') }))
  table.insert(out, pandoc.Div({}, pandoc.Attr('refs')))
  return pandoc.Pandoc(out, doc.meta)
end
