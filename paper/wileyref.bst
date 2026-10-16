% wileyref.bst
% Detailed numeric reference style (IEEE-like order of information, journal abbreviations as given in the .bib):
%   A. Author, B. Author, and C. Author, ``Title,'' J. Abbrev., vol. V, no. N, pp. P--P, Year, doi: DOI.
% References are listed in the order of first citation. DOIs are clickable links.
% The pages field is printed as given (the .bib stores "pp.~916--932" or "Art.~no.~56").

ENTRY
  { author
    title
    journal
    year
    volume
    number
    pages
    howpublished
    doi
  }
  {}
  { label }

STRINGS { s t }

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{f.~}{vv~}{ll}{ jj}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { t "others" =
                { " et~al." * }
                { numnames #2 >
                    { ", and " * t * }
                    { " and " * t * }
                  if$
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {write.entry}
{ 's :=
  "\bibitem{" cite$ * "} " * s * write$
  newline$
}

FUNCTION {article}
{ author format.names
  ", ``" * title * ",'' " * journal *
  volume empty$
    'skip$
    { ", vol.~" * volume * }
  if$
  number empty$
    'skip$
    { ", no.~" * number * }
  if$
  pages empty$
    'skip$
    { ", " * pages * }
  if$
  ", " * year *
  doi empty$
    'skip$
    { ", \href{https://doi.org/" * doi * "}{doi:~\nolinkurl{" * doi * "}}" * }
  if$
  "." *
  write.entry
}

FUNCTION {misc}
{ author empty$
    { title ", " * howpublished * }
    { author format.names ", ``" * title * ",'' " * journal * ", " * year * }
  if$
  "." *
  write.entry
}

FUNCTION {default.type} { misc }

READ

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{99}" write$ newline$
}

EXECUTE {begin.bib}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
