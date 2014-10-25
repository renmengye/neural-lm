
/*
 * Prob.cc --
 *	Functions for handling Probs
 *
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2011 SRI International, 2012-2013 Microsoft Corp.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/Prob.cc,v 1.21 2014-05-22 01:47:40 stolcke Exp $";
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#include "Prob.h"

#include "Array.cc"

const LogP LogP_Zero = -HUGE_VAL;		/* log(0) */
const LogP LogP_Inf = HUGE_VAL;			/* log(Inf) */
const LogP LogP_One = 0.0;			/* log(1) */

const int LogP_Precision = 7;	/* number of significant decimals in a LogP */

const Prob Prob_Epsilon = 3e-06;/* probability sums less than this in
				 * magnitude are effectively considered 0
				 * (assuming they were obtained by summing
				 * LogP's) */

/*
 * parseProb --
 *	Parse a Prob (double prec float) from a string.
 *	We don't enforce that 0 <= prob <= 1, since the values
 *	could be a sum or difference or probs etc.
 *
 * Results:
 *	true if string can be parsed as a float, false otherwise.
 *
 * Side effects:
 *	Result is set to double value if successful.
 *
 */
Boolean
parseProb(const char *str, Prob &result)
{
     double prob;
     if (sscanf(str, "%lf", &prob)) {
	result = prob;
	return true;
     } else {
	return false;
     }
}

/*
 * parseLogP --
 *	Fast parsing of floats representing log probabilities
 *
 * Results:
 *	true if string can be parsed as a float, false otherwise.
 *
 * Side effects:
 *	Result is set to float value if successful.
 *
 */
Boolean
parseLogP(const char *str, LogP2 &result)
{
    const unsigned maxDigits = 8;	// number of decimals in an integer

    const char *cp = str;
    const char *cp0;
    Boolean minus = false;

    if (*cp == '\0') {
	/* empty input */
	return false;
    }

    /*
     * Log probabilties are typically negative values of magnitude > 0.0001,
     * and thus are usually formatted without exponential notation.
     * We parse this type of format using integer arithmetic for speed,
     * and fall back onto scanf() in all other cases.
     * We also use scanf() when there are too many digits to handle with
     * integers.
     * Finally, we also parse +/- infinity values as they are printed by 
     * printf().  These are "[Ii]nf" or "[Ii]nfinity" or "1.#INF".
     */

    /*
     * Parse optional sign
     */
    if (*cp == '-') {
	minus = true;
	cp++;
    } else if (*cp == '+') {
	cp++;
    }
    cp0 = cp;

    unsigned long digits = 0;		// total value of parsed digits
    unsigned long decimals = 1;		// scaling factor from decimal point
    unsigned precision = 0;		// total number of parsed digits

    /*
     * Parse digits before decimal point
     */
    while (isdigit(*cp)) {
	digits = digits * 10 + (*(cp++) - '0');
	precision ++;
    }

    if (*cp == '.') {
	cp++;

	/*
	 * Parse digits after decimal point
	 */
	while (isdigit(*cp)) {
	    digits = digits * 10 + (*(cp++) - '0');
    	    precision ++;
	    decimals *= 10;
	}
    }

    /*
     * If we're at the end of the string then we're done.
     * Otherwise there was either an error or some format we can't
     * handle, so fall back on scanf(), after checking for infinity
     * values.
     */
    if (*cp == '\0' && precision <= maxDigits) {
	result = (minus ? - (LogP2)digits : (LogP2)digits) / (LogP2)decimals;
	return true;
    } else if ((*cp0 == 'i' || *cp0 == 'I' || 
	        (cp0[0] == '1' && cp0[1] == '.' && cp0[2] == '#')) &&
		(strncmp(cp0, "Inf", 3) == 0 || strncmp(cp0, "inf", 3) == 0 ||
		 strncmp(cp0, "1.#INF", 6) == 0))
    {
	result = (minus ? LogP_Zero : LogP_Inf);
	return true;
    } else {
	return (sscanf(str, "%lf", &result) == 1);
    }
}


/* 
 * Codebooks for quantized log probs
 */

Boolean
PQCodebook::read(File &file)
{
    char *line;
    char buffer[10];

    line = file.getline();

    if (!line || sscanf(line, "VQSize %u", &numBins) != 1) {
	file.position() << "missing VQSize spec\n";
	return false;
    }

    for (unsigned i = 0; i < numBins; i ++) {
	binMeans[i] = LogP_Inf;
	binCounts[i] = 0;
    }
     
    line = file.getline();
    if (!line || 
	(sscanf(line, "Codeword Mean %9s", buffer) != 1 &&
	 sscanf(line, "Codword Mean %9s", buffer) != 1) ||
 	strcmp(buffer, "Count") != 0)
    {
	file.position() << "malformed Codeword header\n";
	return false;
    }

    while ((line = file.getline())) {
        unsigned bin;
	double prob;
	unsigned long count;
	if (sscanf(line, "%u %lf %lu", &bin, &prob, &count) != 3) {
	    file.position() << "malformed codeword line\n";
	    return false;
	}

	if (bin >= numBins) {
	    file.position() << "codeword index out of range\n";
	    return false;
	}

	/*
	 * Codebook means are encode as natural logs -- convert to base 10.
	 */
	binMeans[bin] = prob / M_LN10;	
	binCounts[bin] = count;
     }

     return true;
}

Boolean
PQCodebook::write(File &file)
{
    file.fprintf("VQSize %u\n", numBins);
    file.fprintf("Codeword        Mean    Count\n");

    for (unsigned i = 0; i < numBins; i ++) {
	file.fprintf("%8d %20.16lg %12lu\n", 
			i, (double)(binMeans[i] * M_LN10),
			(unsigned long)binCounts[i]);
    }

    return true;
}

LogP2
PQCodebook::getProb(unsigned bin)
{
    if (bin < numBins) {
	return binMeans[bin];
    } else {
	return LogP_Inf;
    }
}

