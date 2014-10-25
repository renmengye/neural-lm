/*
 * version.c --
 *	Print version information
 * 
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2004 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/misc/src/version.c,v 1.5 2013/04/08 20:45:02 victor Exp $";
#endif

#include <stdio.h>

#include "zio.h"
#include "version.h"
#include "SRILMversion.h"

#if defined(_OPENMP) && defined(_MSC_VER)
#include <omp.h>
#endif

void
printVersion(const char *rcsid)
{
	printf("SRILM release %s", SRILM_RELEASE);
#ifndef EXCLUDE_CONTRIB
	printf(" (with third-party contributions)");
#endif /* EXCLUDE_CONTRIB_END */
	printf("\n\nProgram version %s\n", rcsid);
#ifndef NO_ZIO
	printf("\nSupport for compressed files is included.\n");
#else
	printf("\nSupport for gzipped files is included.\n");
#endif
#ifdef HAVE_LIBLBFGS
	printf("Using libLBFGS.\n");
#endif
#ifdef _OPENMP
	printf("Using OpenMP version %d.\n", _OPENMP);
#endif
 	puts(SRILM_COPYRIGHT);
}

