\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{southalabamathesis}
         [2020/04/01 v0.1 University of South Alabama Thesis class]
\RequirePackage{setspace}
\newcounter{subyear}
\setcounter{subyear}{\number\year}
\def\submitted#1{\gdef\@submitted{#1}}
\def\@submittedyear{\ifnum\month>10 \stepcounter{subyear}\thesubyear
  \else\thesubyear\fi}
\def\@submittedmonth{\ifnum\month>10 January\else\ifnum\month>5 November
  \else June\fi\fi}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}}
\def\@copyrightyear{\number\year}
\def\adviser#1{\gdef\@adviser{#1}}
\def\student#1{\gdef\@student{#1}}
\long\def\@abstract{\@latex@error{No \noexpand\abstract given}\@ehc}
\newcommand*{\frontmatter}{
  \pagenumbering{roman}
   }
\newcommand*{\mainmatter}{\pagenumbering{arabic}}
\newcommand*{\makelot}{}
\newcommand*{\makelof}{}
\newcommand*{\makelos}{}
\newcommand*{\makeloa}{}
\newcommand*{\maketoc}{
  % Add ToC to the pdfbookmarks as a section, but not to the ToC itself.
  % only if we have the hyperref command that is necessary
  \ifdefined\pdfbookmark
     \phantomsection
     \pdfbookmark[1]{\contentsname}{TABLE OF CONTENTS}
  \else
  \fi

  \tableofcontents
  \clearpage
}


%% Setup a command to add extra space only if in singlespacing mode
\newcommand*{\extravspace}[1]{\vspace{0in}}
%% Setup a command to set spacing for body text; also used to restore spacing after long tables in document
\newcommand*{\bodyspacing}{
  \doublespacing
}
%% Setup a command to set spacing for long tables, to be used manually in document
\newcommand*{\tablespacing}{
  \singlespacing
}
%% Command inserted into the document immediately after \begin{document} to place all frontmatter
%%   This was formerly named 'begincmd' and called below in \AtBeginDocument{}
%%   However, that is not compatible with package 'hyperref'. \tableofcontents needs
%%   to be declared from within the document itself. 
\newcommand*{\makefrontmatter}{
\bodyspacing
\frontmatter\makesignaturepage\maketitlepage\makededication\makeacknowledgements\maketoc
\makelot\makelof\makeloa\makelos\makeabstract
\clearpage\mainmatter
}
\def\@submitted{\@submittedmonth~\@submittedyear}

\def\@authorreversed{[LastName, FirstName Middle]}
\def\authorreversed#1{\gdef\@authorreversed{#1}}
\def\@expectedgraddate{[Month Year]}
\def\expectedgraddate#1{\gdef\@expectedgraddate{#1}}


\def\@institution{[Institution]}
\def\institution#1{\gdef\@institution{#1}}
\def\@school{[College]}
\def\school#1{\gdef\@school{#1}}
\def\@dept{[Department]}
\def\department#1{\gdef\@dept{#1}}
\def\@degtit{[Degree Title]}
\def\degreetitle#1{\gdef\@degtit{#1}}
\def\@degtyp{[Degree Type]}
\def\degreetype#1{\gdef\@degtyp{#1}}
\def\@degabb{[Degree Abbreviation]}
\def\degreeabbreviation#1{\gdef\@degabb{#1}}
\def\@documenttype{[Document Type]}
\def\documenttype#1{\gdef\@documenttype{#1}}

\def\@documenttype{[Document Type]}
\def\documenttype#1{\gdef\@documenttype{#1}}
\def\@signatories{[Comma-Separated List of Signatories, Signatory 1, Signatory 2]}
\def\signatories#1{\gdef\@signatories{#1}}
\def\@priordegrees{[Comma-Separated List of Prior Degrees, Degree 2]}
\def\priordegrees#1{\gdef\@priordegrees{#1}}

\long\def\acknowledgements#1{\gdef\@acknowledgements{#1}}
\long\def\bio#1{\gdef\@bio{#1}}
\def\dedication#1{\gdef\@dedication{#1}}

\usepackage{pgffor}

\newcommand{\coverpage}[1]{
    \newpage
    \thispagestyle{empty}
    {
        \topskip0pt
        \bfseries
        \vspace*{\fill}
        \begin{center}#1\end{center}
        \vspace*{\fill}
    }
    \clearpage
    \setcounter{page}{\numexpr \thepage-1 \relax}
}


\newcommand{\makesignaturepage}{{
  \thispagestyle{empty}
  \singlespacing
  \begin{center}

    \MakeUppercase{\@institution} \\
    \MakeUppercase{\@school} \\[1.5\baselineskip] 
    \MakeUppercase{\@title} \\[\baselineskip] 
    BY \\[0.8\baselineskip] 
    \@author \\[1.5\baselineskip] 
    \@documenttype \\[1.5\baselineskip] 
    Submitted to the Graduate Faculty of \\
    \@institution \\
    in partial fulfillment of the \\
    requirements for the degree of \\[0.8\baselineskip] 
    \@degtit \\[0.8\baselineskip]
    in \\[0.8\baselineskip] 
    \@degtyp \\[0.8\baselineskip] 
    \@expectedgraddate \\[0.8\baselineskip]
  \end{center}

  \noindent Approved: \hfill Date:\\[1.5\baselineskip]

  \foreach \signatory in \@signatories {
    \noindent
    \hrulefill \\
    \noindent
    \signatory \\[\baselineskip]
  }

  \bodyspacing

  \clearpage
}}




\newcommand{\maketitlepage}{{

  \thispagestyle{empty}
  \setcounter{page}{1}
  \singlespacing
  \begin{center}

    \vspace*{1.625in}

    \MakeUppercase{\@title} \\[4\baselineskip] 
    \@documenttype \\[1.5\baselineskip] 
    Submitted to the Graduate Faculty of \\
    \@institution \\
    in partial fulfillment of the \\
    requirements for the degree of \\[2\baselineskip] 
    \@degtit \\[0.8\baselineskip]
    in \\[0.8\baselineskip] 
    \@degtyp \\[4\baselineskip] 
    by \\
    \@author \\
    \foreach \priordegree in \@priordegrees {
      \priordegree \\
    }
    \@expectedgraddate

  \end{center}

  \bodyspacing
  \clearpage
}}

\newcommand*{\makecopyrightpage}{
  \thispagestyle{empty}
  \setcounter{page}{2}
  \vspace*{0in}
  \begin{center}
    \copyright\ Copyright by \@student, \@copyrightyear. \\
    All rights reserved.
  \end{center}
  \clearpage}

\newcommand{\makeabstract}{{
  \phantomsection
  \addcontentsline{toc}{chapter}{\rm{ABSTRACT}}
  \chapter*{ABSTRACT}
  
  \singlespacing
  \noindent\@abstractopening

  \bodyspacing
  \@abstract
}}

\newcommand{\makebio}{{
  
  
  \phantomsection % makes hyperref recognize this section properly for pdf links
  \coverpage{BIOGRAPHICAL SKETCH}
    
  \addcontentsline{toc}{chapter}{\rm{BIOGRAPHICAL SKETCH}}
  \chapter*{BIOGRAPHICAL SKETCH}

  \@bio

  \clearpage
  }}

\def\makeacknowledgements{
  \ifx\@acknowledgements\undefined
  \else
    \chapter*{ACKNOWLEDGEMENTS}
    \@acknowledgements
    \clearpage
  \fi
  }

\def\makededication{
  \ifx\@dedication\undefined
  \else
    \vspace*{1.5in}
    \begin{center}
      \@dedication
    \end{center}
    \clearpage
  \fi
  }

\DeclareOption{lot}{\renewcommand*{\makelot}{
  \ifdefined\phantomsection
    % makes hyperref recognize this section properly for pdf links
    \phantomsection
  \else
  \fi
  \addcontentsline{toc}{chapter}{\rm{LIST OF TABLES}}\listoftables
  \clearpage
}}
\DeclareOption{lof}{\renewcommand*{\makelof}{
  \ifdefined\phantomsection
    % makes hyperref recognize this section properly for pdf links
    \phantomsection
  \else
  \fi
  \addcontentsline{toc}{chapter}{\rm{LIST OF FIGURES}}\listoffigures
  \clearpage
}}
\DeclareOption{loa}{\renewcommand*{\makeloa}{
  \ifdefined\phantomsection
    % makes hyperref recognize this section properly for pdf links
    \phantomsection
  \else
  \fi
  \addcontentsline{toc}{chapter}{\rm{LIST OF ALGORITHMS}}\listofalg
  \clearpage
}}
\DeclareOption{los}{
  \renewcommand*{\makelos}{
    \phantomsection
    \addcontentsline{toc}{chapter}{\rm{NOMENCLATURE}}
    \printnomenclature
    \clearpage
}}

\DeclareOption{singlespace}{
  \renewcommand*{\bodyspacing}{
  \singlespacing
  }
  %% Add extra space only if in singlespacing mode
  \renewcommand*{\extravspace}[1]{\vspace{#1}}
}
%% Doublespacing is the default for the thesis -- need not be declared.
\DeclareOption{doublespacing}{
  \renewcommand*{\bodyspacing}{
  \doublespacing
  }
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
%% Not necessary to specify the point size - we inherit it from above
%% \LoadClass[12pt]{report}
\LoadClass{report}
%%% Alter LaTeX defaults to try to eliminate all widows and orphans
\clubpenalty=10000
\widowpenalty=10000

%%% try to avoid overfull lines by limiting how far it is okay to exceed the margins
%%% http://www.tex.ac.uk/cgi-bin/texfaq2html?label=overfull
\setlength{\emergencystretch}{2em}

\long\def\abstract#1{\gdef\@abstract{#1}}
\long\def\abstractopening#1{\gdef\@abstractopening{#1}}
%% 'begincmd' no longer used -- insert \makefrontmatter in the document instead. See above. 
%\AtBeginDocument{\begincmd}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
%%%% Tweak float placements

\renewcommand{\topfraction}{0.85}	% max fraction of floats at top
\renewcommand{\bottomfraction}{0.6}	% max fraction of floats at bottom
%   Parameters for TEXT pages (not float pages):
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}     % 2 may work better
\setcounter{dbltopnumber}{2}    % for 2-column pages
\renewcommand{\dbltopfraction}{0.66}	% fit big float above 2-col. text
\renewcommand{\textfraction}{0.15}	% allow minimal text w. figs
%   Parameters for FLOAT pages (not text pages):
\renewcommand{\floatpagefraction}{0.66}	% require fuller float pages
% N.B.: floatpagefraction MUST be less than topfraction !!
\renewcommand{\dblfloatpagefraction}{0.66}	% require fuller float pages
\setlength{\textfloatsep}{30.0pt plus 1.0pt minus 1.0pt}
\setlength{\floatsep}{30.0pt plus 1.0pt minus 1.0pt}

%%% ToC (table of contents) APPEARANCE
%\usepackage[nottoc,notlof,notlot]{tocbibind} % Put the bibliography in the ToC
\usepackage[subfigure]{tocloft} % Alter the style of the Table of Contents
\setcounter{tocdepth}{5}
\setcounter{secnumdepth}{5}
\renewcommand{\cftsecfont}{\rmfamily\mdseries\upshape}
\renewcommand{\cftsecpagefont}{\rmfamily\mdseries\upshape}
\renewcommand{\cftchapfont}{\rmfamily\mdseries\upshape}% No bold!
\renewcommand{\cftchapleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftchappagefont}{\small\normalfont}
\let \savenumberline \numberline
\def \numberline#1{\savenumberline{#1.}}

%%% TODO auto-generate this from the algorithm caption somehow
\newcommand\listalgoname{LIST OF ALGORITHMS\vspace{0.25 in}}
\newlistof{alg}{loalgo}{\listalgoname}
\newcommand{\alg}[1]{%
  \refstepcounter{alg}%
  \addcontentsline{loalgo}{figure}
    {\protect\numberline{\thealg}#1}\par%
}

\renewcommand{\cfttoctitlefont}{\hspace*{\fill}\normalsize\bfseries}
\renewcommand{\cftlottitlefont}{\hspace*{\fill}\normalsize\bfseries}
\renewcommand{\cftloftitlefont}{\hspace*{\fill}\normalsize\bfseries}
\renewcommand{\cftloalgotitlefont}{\hspace*{\fill}\normalsize\bfseries}


\setlength{\cftaftertoctitleskip}{0in}
\setlength{\cftafterloftitleskip}{0in}
\setlength{\cftafterlottitleskip}{0in}
\setlength{\cftafterloalgotitleskip}{0in}
\renewcommand{\cftafterlottitle}{\hspace*{\fill}
  \\[\baselineskip]\mbox{}{\normalfont Table}\hfill{\normalfont Page}}
\renewcommand{\cftaftertoctitle}{\hspace*{\fill}
  \\[\baselineskip]\mbox{}\hfill{\normalfont Page}}
\renewcommand{\cftafterloftitle}{\hspace*{\fill}
  \\[\baselineskip]\mbox{}{\normalfont Figure}\hfill{\normalfont Page}}
\renewcommand{\cftafterloalgotitle}{\hspace*{\fill}
  \\[\baselineskip]\mbox{}{\normalfont Algorithm}\hfill{\normalfont Page}}

\renewcommand*\contentsname{TABLE OF CONTENTS} %change toc name
\renewcommand*\listtablename{LIST OF TABLES\vspace{0.25 in}}
\renewcommand*\listfigurename{LIST OF FIGURES\vspace{0.25 in}}
\renewcommand\bibname{REFERENCES\vspace{0.25 in}}

\providecommand{\e}[1]{\ensuremath{\times 10^{#1}}}

\cftsetindents{figure}{0em}{3em}
\cftsetindents{table}{0em}{3em}
\cftsetindents{chapter}{0em}{2em}
\cftsetindents{section}{2em}{3em}
\cftsetindents{subsection}{4em}{4em}
\cftsetindents{subsubsection}{6em}{5em}

\setlength{\cftchapnumwidth}{7.5em}
\let\oldcftchappresnum\cftchappresnum
\renewcommand*{\cftchappresnum}{CHAPTER \oldcftchappresnum}

%%% HEADERS & FOOTERS
\usepackage{fancyhdr} % This should be set AFTER setting up the page geometry
\pagestyle{fancy} % options: empty , plain , fancy
\renewcommand{\headrulewidth}{0pt} % customize the layout...
\lhead{}\chead{}\rhead{}
\lfoot{}\cfoot{\thepage}\rfoot{}

%%%%Nomenclature

\usepackage[nocfg]{nomencl}
\let\oldnomenclature\nomenclature
\renewcommand{\nomenclature}[2]{\oldnomenclature{\hbox to 0.2\textwidth{#1}#2}{}}
\let\oldnompreamble\nompreamble
\renewcommand{\nompreamble}{\vspace*{0.4in}\oldnompreamble}


%%%%Bibliographic style
\let\oldbibliography\bibliography
\renewcommand{\bibliography}[1]{

    \phantomsection  % makes hyperref recognize this section properly for pdf link
    \coverpage{REFERENCES}
    
    \singlespacing
    
    \addcontentsline{toc}{chapter}{\rm{REFERENCES}}
    \oldbibliography{#1}

    \bodyspacing
}

%%%%Appendix style
\let\oldappendix\appendix
\renewcommand{\appendix}{

    \phantomsection
    \coverpage{APPENDIX}

    \renewcommand{\thesection}{\Alph{section}}
    \addcontentsline{toc}{chapter}{\rm{APPENDIX}}

    \let\oldcftsecpresnum\cftsecpresnum

    \addtocontents{toc}{%
      \protect\setlength{\cftsecnumwidth}{7.5em}%
      \protect\renewcommand*{\protect\cftsecpresnum}{Appendix \oldcftsecpresnum}%
    }

    \titleformat{\section}[display]{\normalfont\bfseries\centering}{Appendix \thesection}{0pt}{}
    \titlespacing*{\section}{0pt}{1\baselineskip}{1\baselineskip}
    \oldappendix
}



%%% Is this not the default?
\raggedright


%%%chapter titles and stuff
\usepackage[largestsep]{titlesec}
\usepackage{chngcntr}
\counterwithout{figure}{chapter}
\counterwithout{figure}{section}
\counterwithout{table}{chapter}
\counterwithout{table}{section}
\titlespacing*{\chapter}{0pt}{1in}{20pt}
\titlespacing*{\section}{0pt}{1\baselineskip}{0pt}
\titlespacing*{\subsection}{0pt}{\ifnum\value{subsection}<2 1\baselineskip \else 0pt \fi}{0pt}
\titlespacing*{\subsubsection}{0pt}{0pt}{0pt}
\titlespacing*{\subsubsubsection}{0pt}{0pt}{0pt}
\titlespacing*{\thebibliography}{0pt}{1in}{20pt}
\renewcommand{\thechapter}{\Roman{chapter}}
\renewcommand{\chaptername}{CHAPTER}
\renewcommand{\thesection}{\arabic{chapter}.\arabic{section}}
%\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}} 
%\renewcommand{\thetable}{\arabic{chapter}.\arabic{figure}}
\titleformat{\chapter}[display]{\normalfont\bfseries\centering}{\chaptername\ \thechapter}{0pt}{}
\titleformat{\section}[hang]{\small\bfseries\centering}{\ul{\thesection \enspace}}{0em}{\ul}
\titleformat{\subsection}[hang]{\small\bfseries\centering}{\thesubsection}{0.33em}{}
\titleformat{\subsubsection}[hang]{\small\bfseries}{\ul{\thesubsubsection \enspace}}{0em}{\ul}



% create an environment that will indent text
% see: http://latex.computersci.org/Reference/ListEnvironments
% 	\raggedright makes them left aligned instead of justified
\newenvironment{indenttext}{
    \begin{list}{}{ \itemsep 0in \itemindent 0in
    \labelsep 0in \labelwidth 0in
    \listparindent 0in
    \topsep 0in \partopsep 0in \parskip 0in \parsep 0in
    \leftmargin 1em \rightmargin 0in
    \raggedright
    }
    \item
  }
  {\end{list}}

% another environment that`s an indented list, with no spaces between items -- if we want multiple items/lines. Useful in tables. Use \item inside the environment.
% 	\raggedright makes them left aligned instead of justified
\newenvironment{indentlist}{
    \begin{list}{}{ \itemsep 0in \itemindent 0in
    \labelsep 0in \labelwidth 0in
    \listparindent 0in
    \topsep 0in \partopsep 0in \parskip 0in \parsep 0in
    \leftmargin 1em \rightmargin 0in
    \raggedright
    }

  }
  {\end{list}}

\setlength{\parindent}{3em}

\usepackage[top=1in, bottom=1.25in, left=1.5in, right=1in]{geometry}
\setlength{\footskip}{0.625in}





\endinput
%% 
%% End of file `southalabamathesis.cls'.
