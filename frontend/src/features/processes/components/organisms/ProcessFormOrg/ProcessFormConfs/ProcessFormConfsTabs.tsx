import * as Separator from "@radix-ui/react-separator";
import * as Tabs from "@radix-ui/react-tabs";
import { ReactNode } from "react";

interface Props<T extends string> {
  tabs: [T, T, ...T[]];
  defaultTab: T;
  onActiveTabChange: (scaling: T) => void;
  children: ReactNode[];
}

export function ProcessFormConfsTabs<T extends string>({
  tabs,
  defaultTab,
  onActiveTabChange,
  children,
}: Props<T>) {
  const tabTriggers = tabs
    .flatMap((tab) => [
      <Tabs.Trigger
        className="text-xs font-bold data-[state=inactive]:opacity-50"
        value={tab}
        key={tab}
      >
        {tab}
      </Tabs.Trigger>,
      <Separator.Root
        className="mx-1.5 bg-white data-[orientation=vertical]:h-full data-[orientation=vertical]:w-px"
        orientation="vertical"
        key={`${tab} - Separator.Root`}
      />,
    ])
    .slice(0, -1);
  const tabContents = tabs.map((tab, i) => (
    <Tabs.Content value={tab} key={tab}>
      {children[i]}
    </Tabs.Content>
  ));

  return (
    <Tabs.Root
      className="space-y-4"
      defaultValue={defaultTab}
      onValueChange={(value) => onActiveTabChange(value as T)}
    >
      <Tabs.List className="flex h-3 items-center justify-center">
        {tabTriggers}
      </Tabs.List>
      {tabContents}
    </Tabs.Root>
  );
}
