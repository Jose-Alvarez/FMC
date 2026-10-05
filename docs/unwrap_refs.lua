-- unwrap_refs.lua: runs after --citeproc and removes the <div> wrappers of the reference list,
-- so that README.md contains plain Markdown paragraphs.
function Div(d)
  if d.identifier == 'refs' or d.classes:includes('csl-entry') then
    return d.content
  end
end
