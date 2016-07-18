#!/usr/bin/perl
# script para juntar los topicos y texto del tweet en una sola linea
# ejecucion 
# perl -Mutf8 -CS script.pl > TASSTrain.txt
use XML::XPath;
use Date::Parse;
use Date::Format;

my $file = 'tass_2015/general-tweets-test.xml'; 
# general-tweets-train-tagged.xml
# general-tweets-test1k-tagged.xml
my $xp = XML::XPath->new(filename => $file);
my $count = 1;
foreach my $entry ($xp->find('//tweet')->get_nodelist){
    print "\"";
    my $deit = $entry->find('date'); 
    $time = str2time($deit);
    #$time = $class->parse_datetime($deit);
    #$string = $class->format_datetime($time); # Format as GMT ASCII time
    #print $time;
    #print ctime(time)
    print time2str("%a %b %e %T +0000 %Y", $time);
    #print time2str("%C",$time);
    print "\\\\\\\\\\\\";
    print $entry->find('tweetid');
    print "\\\\\\\\\\\\";
    my $str = $entry->find('content')->string_value;
    $str =~ s/\R//g;
    print $str ;
    print "\\\\\\\\\\\\";
    foreach my $topic ($entry->find('topics/topic')->get_nodelist){
	print $topic->string_value . ",";
    }
    print "\"\n";
    $count++;
}
